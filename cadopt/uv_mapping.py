import numpy as np
import xatlas
from typing import Tuple, Optional, List, Dict

def compute_patch_surface_areas(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    face_indices: np.ndarray
) -> Dict[int, float]:
    """Computes 3D surface area for each unique CAD patch ID."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    patch_areas = {}
    unique_fids = np.unique(face_indices)
    for fid in unique_fids:
        mask = (face_indices == fid)
        patch_areas[int(fid)] = float(np.sum(tri_areas[mask]))
        
    return patch_areas

def apply_xatlas_uvs(
    vertices: np.ndarray, 
    normals: Optional[np.ndarray], 
    faces: np.ndarray,
    face_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Parametrize the mesh using xatlas, "weaponizing" CAD patch boundaries 
    to force semantic seams exactly where the physical parts meet.
    """
    print(f"[LOG] UV: Starting Semantic UV Unwrapping on {len(vertices)} vertices...")
    atlas = xatlas.Atlas()
    patch_global_indices = []
    
    if face_indices is not None:
        unique_fids = np.unique(face_indices)
        print(f"[LOG] UV: Found {len(unique_fids)} unique CAD patches. Split-unwrapping...")
        
        for fid in unique_fids:
            f_mask = (face_indices == fid)
            sub_faces = faces[f_mask]
            
            # Map sub-faces to local indices
            sub_v_idx = np.unique(sub_faces)
            v_map = {old: new for new, old in enumerate(sub_v_idx)}
            local_faces = np.vectorize(v_map.get)(sub_faces).astype(np.uint32)
            
            sub_v = vertices[sub_v_idx].astype(np.float32)
            sub_vn = normals[sub_v_idx].astype(np.float32) if normals is not None else None
            
            atlas.add_mesh(sub_v, local_faces, sub_vn)
            patch_global_indices.append(sub_v_idx)
    else:
        atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32), 
                       normals.astype(np.float32) if normals is not None else None)
        patch_global_indices.append(np.arange(len(vertices)))
    
    atlas.generate()
    
    # Reconstruct unified mesh from atlas with reconciled indices
    vmapping_global, indices, uvs = _reconstruct_multi_mesh_atlas(atlas, patch_global_indices)
    
    new_vertices = vertices[vmapping_global]
    new_faces = indices
    new_uvs = uvs
    new_normals = normals[vmapping_global] if normals is not None else None
    
    print(f"[LOG] UV: Semantic unwrap complete. {len(new_vertices)} vertices, {len(new_faces)} tris.")
    return new_vertices, new_normals, new_faces, new_uvs

def apply_xatlas_udim_uvs(
    vertices: np.ndarray,
    normals: Optional[np.ndarray],
    faces: np.ndarray,
    face_indices: Optional[np.ndarray] = None,
    uv_settings: Optional[object] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Parametrize mesh using UDIM multi-tile packing, grouping CAD patches 
    by surface area into separate tiles (Tile 1001, 1002, 1003, ...).
    """
    mode = getattr(uv_settings, "mode", "SingleTile")
    if mode == "SingleTile" or face_indices is None:
        return apply_xatlas_uvs(vertices, normals, faces, face_indices=face_indices)

    max_cols = getattr(uv_settings, "max_udim_cols", 10)
    thresholds = getattr(uv_settings, "bucket_thresholds_mm2", [1000.0, 100.0, 10.0])

    print(f"[LOG] UDIM: Starting Multi-Tile UDIM Unwrapping (mode={mode}) on {len(vertices)} vertices...")

    patch_areas = compute_patch_surface_areas(vertices, faces, face_indices)
    unique_fids = np.array(list(patch_areas.keys()))
    areas = np.array([patch_areas[fid] for fid in unique_fids])

    # Assign patches to tile buckets based on selected UDIM mode
    tile_buckets: List[List[int]] = []

    if mode == "UDIM-Bucketed":
        # Threshold-based bucketing: area >= T0 -> Tile 0, T1 <= area < T0 -> Tile 1, etc.
        sorted_thresholds = sorted(thresholds, reverse=True)
        nb_buckets = len(sorted_thresholds) + 1
        tile_buckets = [[] for _ in range(nb_buckets)]

        for fid, area in patch_areas.items():
            assigned = False
            for idx, thresh in enumerate(sorted_thresholds):
                if area >= thresh:
                    tile_buckets[idx].append(fid)
                    assigned = True
                    break
            if not assigned:
                tile_buckets[-1].append(fid)

    else:  # UDIM-Auto
        # Area-ranked dynamic grouping: sort patches descending by surface area
        sorted_indices = np.argsort(-areas)
        sorted_fids = unique_fids[sorted_indices]

        # Allocate patches into tiles up to ~80% surface area capacity per tile
        total_area = np.sum(areas)
        target_area_per_tile = max(10.0, total_area / 4.0)

        current_bucket = []
        current_area = 0.0

        for fid in sorted_fids:
            a = patch_areas[fid]
            if current_bucket and (current_area + a > target_area_per_tile):
                tile_buckets.append(current_bucket)
                current_bucket = [fid]
                current_area = a
            else:
                current_bucket.append(fid)
                current_area += a

        if current_bucket:
            tile_buckets.append(current_bucket)

    # Filter out empty buckets
    tile_buckets = [b for b in tile_buckets if len(b) > 0]
    print(f"[LOG] UDIM: Allocated {len(unique_fids)} CAD patches into {len(tile_buckets)} UDIM tile buckets.")

    all_verts = []
    all_normals = []
    all_faces = []
    all_uvs = []
    v_offset = 0

    for tile_idx, patch_ids in enumerate(tile_buckets):
        u_col = tile_idx % max_cols
        v_row = tile_idx // max_cols
        udim_number = 1001 + u_col + 10 * v_row

        # Filter faces belonging to these patch IDs
        mask = np.isin(face_indices, patch_ids)
        sub_faces = faces[mask]

        if len(sub_faces) == 0:
            continue

        # Extract local vertices and faces for this tile bucket
        sub_v_idx = np.unique(sub_faces)
        v_map = {old: new for new, old in enumerate(sub_v_idx)}
        local_faces = np.vectorize(v_map.get)(sub_faces).astype(np.uint32)

        sub_v = vertices[sub_v_idx].astype(np.float32)
        sub_vn = normals[sub_v_idx].astype(np.float32) if normals is not None else None

        # Unwrapping this tile bucket with xatlas
        atlas = xatlas.Atlas()

        # Split sub-patches for zero seam distortion
        patch_global_indices = []
        for fid in patch_ids:
            f_mask = (face_indices[mask] == fid)
            if not np.any(f_mask):
                continue
            p_sub_faces = local_faces[f_mask]
            p_v_idx = np.unique(p_sub_faces)
            p_v_map = {old: new for new, old in enumerate(p_v_idx)}
            p_local_faces = np.vectorize(p_v_map.get)(p_sub_faces).astype(np.uint32)

            p_v = sub_v[p_v_idx]
            p_vn = sub_vn[p_v_idx] if sub_vn is not None else None

            atlas.add_mesh(p_v, p_local_faces, p_vn)
            patch_global_indices.append(sub_v_idx[p_v_idx])

        atlas.generate()

        # Reconstruct tile geometry
        vmapping_global, indices, uvs = _reconstruct_multi_mesh_atlas(atlas, patch_global_indices)

        # Shift UV coordinates to assigned UDIM tile offset (u_col, v_row)
        shifted_uvs = uvs.copy()
        shifted_uvs[:, 0] += u_col
        shifted_uvs[:, 1] += v_row

        t_verts = vertices[vmapping_global]
        t_normals = normals[vmapping_global] if normals is not None else None

        all_verts.append(t_verts)
        if t_normals is not None:
            all_normals.append(t_normals)
        all_faces.append(indices + v_offset)
        all_uvs.append(shifted_uvs)

        v_offset += len(t_verts)

        print(f"  -> Tile {udim_number} (Col {u_col}, Row {v_row}): {len(patch_ids)} CAD patches, {len(shifted_uvs):,} UV coords.")

    final_vertices = np.concatenate(all_verts)
    final_faces = np.concatenate(all_faces)
    final_uvs = np.concatenate(all_uvs)
    final_normals = np.concatenate(all_normals) if len(all_normals) > 0 else None

    print(f"[LOG] UDIM: Multi-tile unwrapping complete. {len(final_vertices):,} vertices, {len(final_faces):,} tris across {len(tile_buckets)} UDIM tiles.")

    return final_vertices, final_normals, final_faces, final_uvs

def _reconstruct_multi_mesh_atlas(atlas: xatlas.Atlas, patch_global_indices: List[np.ndarray]):
    """Combines multiple charts back into a single geometry set with reconciled global indices."""
    all_vmapping_reconciled = []
    all_indices = []
    all_uvs = []
    v_offset = 0
    
    for i in range(atlas.mesh_count):
        vm_local, ind, uv = atlas.get_mesh(i)
        
        # KEY FIX: Map local vmapping (index into sub_v) back to global index (index into root vertices)
        sub_v_idx = patch_global_indices[i]
        vm_global = sub_v_idx[vm_local]
        
        all_vmapping_reconciled.append(vm_global)
        all_indices.append(ind + v_offset)
        all_uvs.append(uv)
        v_offset += len(vm_local)
        
    return (
        np.concatenate(all_vmapping_reconciled),
        np.concatenate(all_indices),
        np.concatenate(all_uvs)
    )
