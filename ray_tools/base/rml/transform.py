import pandas as pd
import numpy as np

def translation_matrix(tx: float, ty: float, tz: float)-> np.ndarray:
    matrix = [
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]
    ]
    return np.array(matrix)

def rotation_matrix_x(alpha: float)-> np.ndarray:
    rotation_alpha = [
        [1, 0, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha), 0],
        [0, np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1]
    ]
    return np.array(rotation_alpha)

def rotation_matrix_y(beta: float)-> np.ndarray:
    rotation_beta = [
        [np.cos(beta), 0, np.sin(beta), 0],
        [0, 1, 0, 0],
        [-np.sin(beta), 0, np.cos(beta), 0],
        [0, 0, 0, 1]
    ]
    return np.array(rotation_beta)

def rotation_matrix_z(gamma: float)-> np.ndarray:
    rotation_gamma = [
        [np.cos(gamma), -np.sin(gamma), 0, 0],
        [np.sin(gamma), np.cos(gamma), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    return np.array(rotation_gamma)

def apply_rigid_transform(
    base_pos: np.ndarray,
    base_dirs: np.ndarray,
    errs: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    # 1) Build homogeneous basis (s, vx, vy, vz)
    s  = np.concatenate([base_pos, [1.0]])
    vx = np.concatenate([base_dirs[0], [0.0]])
    vy = np.concatenate([base_dirs[1], [0.0]])
    vz = np.concatenate([base_dirs[2], [0.0]])
    
    R_W = np.stack([vx, vy, vz, [0,0,0,1]], axis=1)
    T_W = translation_matrix(s[0], s[1], s[2]) @ R_W

    # 2) Local translation vector (include Z!)
    tx = errs.get('translationXerror', 0.0)
    ty = errs.get('translationYerror', 0.0)
    tz = errs.get('translationZerror', 0.0)
    t_local = np.array([tx, ty, tz])

    # 3) Local rotation angles (convert mrad → rad)
    alpha = np.array([
        errs.get('rotationXerror', 0.0),
        errs.get('rotationYerror', 0.0),
        errs.get('rotationZerror', 0.0)
    ], dtype=float) * 1e-3  # now in radians

    # 4) Build the local transform T_local = Rz·Ry·Rx · T(t)
    T_local = (
        rotation_matrix_z(alpha[2]) @
        rotation_matrix_y(alpha[1]) @
        rotation_matrix_x(alpha[0]) @
        translation_matrix(t_local[0], t_local[1], t_local[2])
    )

    # 5) Apply
    s_new  = T_W @ T_local @ np.array([0,0,0,1])
    vx_new = T_W @ T_local @ np.array([1,0,0,0])
    vy_new = T_W @ T_local @ np.array([0,1,0,0])
    vz_new = T_W @ T_local @ np.array([0,0,1,0])

    new_pos  = s_new[:3]
    new_dirs = np.stack([vx_new[:3], vy_new[:3], vz_new[:3]])

    return new_pos, new_dirs

def parse_error_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV containing file1 (with errors) and file2 (baseline) data.
    Returns a DataFrame indexed by 'Object'.
    """
    df = pd.read_csv(csv_path)
    df.set_index('Object', inplace=True)
    return df

def test_zero_errors():
    # Ensure identity transform when errors are zero.
    base_pos = np.array([1.0, 2.0, 3.0], dtype=float)
    base_dirs = np.eye(3, dtype=float)
    errs = dict.fromkeys(['translationX','translationY','translationZ','rotationX','rotationY','rotationZ'], 0.0)
    new_pos, new_dirs = apply_rigid_transform(base_pos, base_dirs, errs)
    assert np.allclose(new_pos, base_pos), f"Zero error pos fails: {new_pos}"
    assert np.allclose(new_dirs, base_dirs), f"Zero error dirs fails: {new_dirs}"
    print("Zero-errors test passed.")

if __name__ == '__main__':
    df = parse_error_csv("rml_requested_param_differences.csv")
    print(df.head())
    print(df.columns)

    test_zero_errors()

    for index, row in df.iterrows():
        print(f"Running test for {row.name}")

        base_pos = np.array([
            row["worldPosition_x_file2"],
            row["worldPosition_y_file2"],
            row["worldPosition_z_file2"],
        ], dtype=float)

        base_dirs = np.array([
            [row["worldXdirection_x_file2"], row["worldXdirection_y_file2"], row["worldXdirection_z_file2"]],
            [row["worldYdirection_x_file2"], row["worldYdirection_y_file2"], row["worldYdirection_z_file2"]],
            [row["worldZdirection_x_file2"], row["worldZdirection_y_file2"], row["worldZdirection_z_file2"]],
        ], dtype=float)

        errs = {
            'translationX': row['translationXerror_file1'],
            'translationY': row['translationYerror_file1'],
            'translationZ': row['translationZerror_file1'],
            'rotationX': row['rotationXerror_file1'],
            'rotationY': row['rotationYerror_file1'],
            'rotationZ': row['rotationZerror_file1'],
        }

        new_pos, new_dirs = apply_rigid_transform(base_pos, base_dirs, errs)

        expected_pos = np.array([
            row["worldPosition_x_file1"],
            row["worldPosition_y_file1"],
            row["worldPosition_z_file1"],
        ], dtype=float)

        expected_dirs = np.array([
            [row["worldXdirection_x_file1"], row["worldXdirection_y_file1"], row["worldXdirection_z_file1"]],
            [row["worldYdirection_x_file1"], row["worldYdirection_y_file1"], row["worldYdirection_z_file1"]],
            [row["worldZdirection_x_file1"], row["worldZdirection_y_file1"], row["worldZdirection_z_file1"]],
        ], dtype=float)

        if not np.allclose(new_pos, expected_pos, atol=1e-5):
            print(f"[FAIL] {row.name}: Position mismatch")
            print(f"Expected: {expected_pos}, Got: {new_pos}")
        elif not np.allclose(new_dirs, expected_dirs, atol=1e-5):
            print(f"[FAIL] {row.name}: Direction mismatch")
            print(f"Expected:\n{expected_dirs}\nGot:\n{new_dirs}")
        else:
            print(f"[OK] {row.name}")


