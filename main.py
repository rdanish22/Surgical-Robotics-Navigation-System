import numpy as np
from Algorithms import solve_pivot_calibration
from Algorithms import registration_algo
from Algorithms import compute_expected_C
from Cartesian_Math import frame_transformation
from IO import read_calbody
from IO import read_calreadings
from IO import read_empivot_data
from IO import read_optpivot_data
from IO import read_output_data
from IO import write_files


def check_error(EM_dimple_position, optical_dimple_position, C_expected, output_path, letter):
    check_EM_pos, check_opt_pos, check_C = read_output_data(output_path)

    print("Output comparison for "f"file-{letter}" + "\n")
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



def main():

    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    base_path = "PA1StudentData"

    for letter in letters:
        type = 'debug'
        if ord(letter) > ord('g'):
            type = 'unknown'
        
        file_name = f"{base_path}/pa1-{type}-{letter}"

        calbody_path = f"{file_name}-calbody.txt"
        calreadings_path = f"{file_name}-calreadings.txt"
        empivot_path = f"{file_name}-empivot.txt"
        optpivot_path = f"{file_name}-optpivot.txt"

        # question 4
        d_markers, a_markers, c_markers, N_C = read_calbody(calbody_path)
        D_frames, A_frames, C_frames, N_frames = read_calreadings(calreadings_path)
        C_expected_all_frames = compute_expected_C(d_markers, a_markers, c_markers, D_frames, A_frames)
        C_expected_all_frames = np.concatenate(C_expected_all_frames, axis=0)

        # question 5
        frames_data = read_empivot_data(empivot_path)
        EM_dimple_position = solve_pivot_calibration(frames_data)

        # question 6
        Dframes_data, Hframes_data = read_optpivot_data(optpivot_path)
        F_D = registration_algo(d_markers, Dframes_data[0].T)
        F_D_inv = F_D.inverse()
        G = frame_transformation(Hframes_data, F_D_inv)
        optical_dimple_position = solve_pivot_calibration(G)

        # check error
        if type == 'debug':
            output_path = f"{file_name}-output1.txt"
            check_error(EM_dimple_position, optical_dimple_position, C_expected_all_frames, output_path, letter)

        #make output file
        write_files(type, letter, N_C, N_frames, EM_dimple_position, optical_dimple_position, C_expected_all_frames)


if __name__ == '__main__':
    main()
