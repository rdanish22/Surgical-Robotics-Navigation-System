import numpy as np
from Algorithms import solve_pivot_calibration
from Algorithms import registration_algo
from Algorithms import compute_expected_C
from Algorithms import distortion_correction
from Algorithms import get_bernstein_coeff
from Algorithms import compute_F_matrix
from Algorithms import compute_em_fudicials
from Cartesian_Math import frame_transformation
from IO import read_calbody
from IO import read_calreadings
from IO import read_empivot_data
from IO import read_em_fiducial_data
from IO import read_ct_fiducials
from IO import read_em_nav_data
from IO import read_optpivot_data
from IO import write_files_pa2
from IO import read_output_data
from IO import read_output2_data

def check_error(tips, EM_dimple_position, optical_dimple_position, C_expected, output_path1, output_path2, letter):
    check_tips = read_output2_data(output_path2)
    check_EM_pos, check_opt_pos, check_C = read_output_data(output_path1)

    print("Output 1 comparison for "f"file-{letter}" + "\n")
    print("Error Percentage")
    print(np.mean(np.abs(EM_dimple_position - check_EM_pos) / check_EM_pos) * 100)
    print(np.mean(np.abs(optical_dimple_position - check_opt_pos) / check_opt_pos) * 100)
    print(np.mean(np.abs(C_expected - check_C) / check_C) * 100)
    print("\n")

    print("Max Difference")
    print(np.abs(np.max(EM_dimple_position - check_EM_pos)))
    print(np.abs(np.max(optical_dimple_position - check_opt_pos)))
    print(np.abs((np.max(C_expected - check_C))))
    print("\n")

    print("Output 2 comparison for "f"file-{letter}" + "\n")
    print("Error Percentage")
    print(np.mean(np.abs(tips.squeeze() - check_tips) / check_tips) * 100)

    print("Max Difference")
    print(np.abs(np.max(tips.squeeze() - check_tips)))

    print("Euclidean Distance")
    print(np.linalg.norm(tips.squeeze() - check_tips))

    print("\n")

def main():
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    base_path = "PA2StudentData"

    for letter in letters:
        type = 'debug'
        if ord(letter) > ord('f'):
            type = 'unknown'
        
        file_name = f"{base_path}/pa2-{type}-{letter}"

        calbody_path = f"{file_name}-calbody.txt"
        calreadings_path = f"{file_name}-calreadings.txt"
        empivot_path = f"{file_name}-empivot.txt"
        optpivot_path = f"{file_name}-optpivot.txt"
        em_fiducials_file = f"{file_name}-em-fiducialss.txt"
        ct_fiducials_file = f"{file_name}-ct-fiducials.txt"
        em_nav_file = f"{file_name}-EM-nav.txt"

        d_markers, a_markers, c_markers, N_C = read_calbody(calbody_path)
        D_frames, A_frames, C_frames, N_frames = read_calreadings(calreadings_path)
        C_frames = np.concatenate(C_frames, axis=0)

        #question 1
        C_expected_all_frames = compute_expected_C(d_markers, a_markers, c_markers, D_frames, A_frames)
        C_expected_all_frames = np.concatenate(C_expected_all_frames, axis=0)

        # calculate mins and maxs for bounding boxes
        C_exp_min = np.min(C_expected_all_frames, axis=0)
        C_exp_max = np.max(C_expected_all_frames, axis=0)
        C_bounded = (C_expected_all_frames - C_exp_min) / (C_exp_max - C_exp_min)
        C_min = np.min(C_frames, axis=0)
        C_max = np.max(C_frames, axis=0)

        # question 2
        bernstein_C = compute_F_matrix(C_frames, C_min, C_max)
        coeffs = get_bernstein_coeff(bernstein_C, C_bounded)

        frames_data = read_empivot_data(empivot_path)
        EM_distorted_tip, EM_distorted_dimple = solve_pivot_calibration(frames_data)

        # question 3
        EMpiv_undistorted = distortion_correction(coeffs, frames_data, C_min, C_max, C_exp_min, C_exp_max)
        EM_tip_pos, EM_dimple_pos = solve_pivot_calibration(EMpiv_undistorted)
        EM_tip_pos = EM_tip_pos.reshape(3, 1)

        # question 4
        em_fiducial_frames = read_em_fiducial_data(em_fiducials_file)
        em_fiducials_undistorted = distortion_correction(coeffs, em_fiducial_frames, C_min, C_max, C_exp_min, C_exp_max)
        fudicial_point_locations = compute_em_fudicials(em_fiducials_undistorted, EMpiv_undistorted, EM_tip_pos)

        # question 5
        ct_fiducials = read_ct_fiducials(ct_fiducials_file)
        F_reg = registration_algo(fudicial_point_locations, ct_fiducials)

        # question 6
        em_nav_frames, NG, N_frames_em_nav = read_em_nav_data(em_nav_file)
        em_nav_undistorted = distortion_correction(coeffs, em_nav_frames, C_min, C_max, C_exp_min, C_exp_max)
        nav_fudicial_points = compute_em_fudicials(em_nav_undistorted, EMpiv_undistorted, EM_tip_pos).T

        em_nav_tips = []
        for i in range(len(nav_fudicial_points)):
            transformed_point = frame_transformation(nav_fudicial_points[i].reshape(3, 1), F_reg)
            em_nav_tips.append(transformed_point)

        rounded_tips = np.round(em_nav_tips, 2)

        Dframes_data, Hframes_data = read_optpivot_data(optpivot_path)
        F_D = registration_algo(d_markers, Dframes_data[0].T)
        F_D_inv = F_D.inverse()
        G = frame_transformation(Hframes_data, F_D_inv)
        optical_tip, optical_dimple_position = solve_pivot_calibration(G)

        if type == 'debug':
            output_path1 = f"{file_name}-output1.txt"
            output_path2 = f"{file_name}-output2.txt"
            check_error(rounded_tips, EM_distorted_dimple, optical_dimple_position, C_expected_all_frames, output_path1, output_path2, letter)

        write_files_pa2(type, letter, N_C, N_frames, EM_distorted_dimple, optical_dimple_position, C_expected_all_frames, N_frames_em_nav, rounded_tips)


if __name__ == '__main__':
    main()
