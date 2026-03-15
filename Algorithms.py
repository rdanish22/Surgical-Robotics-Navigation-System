import numpy as np
from math import comb
from Frame import Frame
from Cartesian_Math import frame_transformation

def registration_algo(A, B):
    """
    Answer to question 2, computes a frame from a point cloud to point cloud registration algortithm.

    The algorithm used is the direct technique to compute R using SVD outlined in slide 9, panel 17 
    in the Point cloud to point cloud rigid transformations lecture slides. The way to compute p is
    from the equation given in slide 2, panel 4 of the same lecture.

    Parameters:
    - A: Array of float64 that represents the starting point cloud
    - B: Array of float64 that represents the desired point cloud to transform into
    
    Returns:
    - Frame that gives transformation from point cloud to point cloud
    """

    # Calculate center of both point clouds
    A_cloud_center = np.mean(A, axis=1, keepdims=True)
    B_cloud_center = np.mean(B, axis=1, keepdims=True)

    # translate all points relative to the center
    A_centered = A - A_cloud_center
    B_centered = B - B_cloud_center

    # Start of SVD method from slides 

    # Step 1, find H
    H = A_centered @ B_centered.T

    # Step 2, Compute SVD of H
    U, _, V_t = np.linalg.svd(H)

    # Step 3, Find R
    R = V_t.T @ U.T

    # Step 4, Verify that det(R) = 1, if not correct V_t and recompute R
    if np.linalg.det(R) < 0:
        V_t[-1, :] *= -1
        R = V_t.T @ U.T

    #Compute p
    p = B_cloud_center - (R @ A_cloud_center)
    
    return Frame(R, p)

def pivot_calibration(frames):
    """
    Answer to question 3, computes the pivot calibration of a pointer.

    The method this function uses to compute pivot calibration is least squares as outlined in 
    slide 28, panel 61 of the Cartesian Coordinates, Points, and Transformations lecture slide.

    Parameters:
    - Rs: Array of rotation matrices
    - ps: Array of p matrices
    
    Returns:
    - A 2D array with the solution to the least squares problem
    """
    R_I = []
    p_neg = []

    # process the rotation and position matrix to be in the form need for pivot calibration
    for frame in frames:
        R_I.append(np.hstack((frame.R, -1 * np.eye(3))))
        p_neg.append(-1 * frame.P)

    # Concatenate all the rows in A and b to make them 2 dimentional
    R_I = np.concatenate(R_I, axis=0)
    p_neg = np.concatenate(p_neg, axis=0)

    # solve and return the least squares problem R * b_tip - b_pivot = p
    return np.linalg.lstsq(R_I, p_neg)[0]

def solve_pivot_calibration(frames_data):
    """
    Helper function for question 5, and as an extension, question 6.

    Performs pivot calibration of a probe in EM tracker base coordinates from data frames.
    
    Parameters:
    - frames_data: List of numpy arrays, each representing the 3D coordinates of markers.
    
    Returns:
    - Estimated dimple position relative to the EM tracker base coordinates.
    """

    # get the first frame to be used to define a local probe coordinate system
    first_frame = frames_data[0]
    
    # compute midpoint of observed points
    G_0 = np.mean(first_frame, axis=1, keepdims=True)
    
    # translate observations relative to midpoint
    g_j = first_frame - G_0
    
    # define arrays to be used to solve pivot calibration least squares problem of the form Ax = b
    frames = []
    
    # for each frame in the pivot data frame
    for frame in frames_data:

        # compute transformation F_g such that G_j(frame) = F_g * g_j
        F_g = registration_algo(g_j, frame)
        
        # make a vector of Frames
        frames.append(F_g)
    
    # perform pivot calibration
    x = pivot_calibration(frames)
    x = x.squeeze()
    
    # isolate tip and dimple coordinates from pivot calibration results which are of the form [P_tip | P_dimple].T
    P_tip = x[:3]
    P_dimple = x[3:]

    return P_tip, P_dimple

def compute_expected_C(d_markers, a_markers, c_markers, D_frames, A_frames):
    """
    Helper function for question 4

    Computes C_expected for a distorted calibration data set
    
    Parameters:
    - calreadings_path: File path for distorted calibration data set
    - d_markers: Array of coordinates for optical markers on EM base
    - a_markers: Array of coordinates for optical markers on calibration object
    - c_markers: Array of coordinates for EM markers on calibration object
    
    Returns:
    - calculated C_expected for all data frames
    """

    C_expected_all_frames = []

    # For each frame, compute C_expected
    for frame_idx in range(len(D_frames)):

        # isolate the optical tracker coordinates of optical markers on EM base in one frame
        D_frame = np.array(D_frames[frame_idx]).T
        # isolate the optical tracker coordinates of optical markers on calibration object in one frame
        A_frame = np.array(A_frames[frame_idx]).T

        # Compute F_D and F_A using point cloud registration
        F_D = registration_algo(d_markers, D_frame)
        F_A = registration_algo(a_markers, A_frame)
    
        # Apply the transformations to c_markers to compute expected positions
        F_D_inv = F_D.inverse()
        F_A_C = frame_transformation(c_markers, F_A)
        C_expected = frame_transformation(F_A_C, F_D_inv)

        # Store expected positions for this frame
        C_expected_all_frames.append(C_expected.T)

    return np.array(C_expected_all_frames)

def bernstein_polynomial(n, i, u):
    """
    Computes the Bernstein polynomial basis function value.

    Parameters:
    - n: Degree of the polynomial.
    - i: Term index (0 <= i <= n).
    - u: Value(s) at which to evaluate (can be array).

    Returns:
    - B: Value(s) of the Bernstein polynomial.
    """
    return comb(n, i) * ((1 - u) ** (n - i)) * (u ** i)

def compute_F_matrix(q, qmin, qmax, n=5):
    """
    Computes the F matrix for the Bernstein polynomial basis functions.

    Parameters:
    - q: Distorted data array of shape (3, N).
    - qmin: Minimum of distorted data
    - qmax: Maximum of distorted data
    - n: Degree of the polynomials (e.g., 5).

    Returns:
    - F: Feature matrix of shape (N, (n+1)^3).
    """

    u = (q - qmin) / (qmax - qmin)
    
    N = u.shape[0]
    F = np.zeros((N, (n+1)**3))

    for n_i in range(N):
        idx = 0
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    F[n_i][idx] = bernstein_polynomial(n, i, u[n_i][0]) * bernstein_polynomial(n, j, u[n_i][1]) * bernstein_polynomial(n, k, u[n_i][2])
                    idx += 1
    return F

def get_bernstein_coeff(F, p):

    """
    Computes the distortion correction coefficients using Bernstein polynomials.

    Parameters:
    - F: Distorted data array of shape (3, N).
    - p: Ground truth data array of shape (3, N).

    Returns:
    - coeffs: Coefficients of the distortion correction function.
    """

    coeffs = np.linalg.lstsq(F, p)[0]

    return coeffs

def distortion_correction(coeffs, q, qmin, qmax, cmin, cmax):
    p_corrected = np.zeros_like(q)
    q = np.array(q)
    q = np.transpose(q, (0, 2, 1))
    q = np.concatenate(q, axis=0)

    F = compute_F_matrix(q, qmin, qmax)
    
    for dim in range(3):
        p_corrected[:, dim, :] = np.array(F @ coeffs[:, dim]).reshape(p_corrected[:, dim, :].shape)

    cmax = cmax.reshape(3, 1)
    cmin = cmin.reshape(3, 1)
    p_corrected = (p_corrected * (cmax - cmin)) + cmin
    
    return p_corrected


def compute_em_fudicials(EM_frames, EM_piv_frames, pointer_tip):

    # get the first frame to be used to define a local probe coordinate system
    first_frame = EM_piv_frames[0]
    
    # compute midpoint of observed points
    G_0 = np.mean(first_frame, axis=1, keepdims=True)
    
    # translate observations relative to midpoint
    g_j = first_frame - G_0

    fudicial_all_frames = []

    # For each frame, compute C_expected
    for frame in EM_frames:

        F_g = registration_algo(g_j, frame)
        fudicial_point = frame_transformation(pointer_tip, F_g)
        fudicial_all_frames.append(fudicial_point.flatten())

    return np.array(fudicial_all_frames).T

