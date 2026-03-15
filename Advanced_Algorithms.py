import numpy as np
from Frame import Frame
from KDTreeNode import KDNode

def registration_algo(A, B):
    """
    Computes a frame from a point cloud to point cloud registration algortithm.

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
    U, S, V_t = np.linalg.svd(H)

    # Step 3, Find R
    R = V_t.T @ U.T

    # Step 4, Verify that det(R) = 1, if not correct V_t and recompute R
    if np.linalg.det(R) < 0:
        V_t[-1, :] *= -1
        R = V_t.T @ U.T

        # check if any singular value is 0
        singular_values = np.diagonal(S)
        if np.any(singular_values != 0):
            print("Determinant is -1 and none of the singular values are 0, cannot compute rotation")

    #Compute p
    p = B_cloud_center - (R @ A_cloud_center)
    
    return Frame(R, p)

def compute_A_tip_in_B(N_samples, sample_a_markers, sample_b_markers, markers_A, markers_B, tip_A):
    """
    Algorithm to compute the tips of pointer A in rigid body B's coordinate frame as outlined in the hint on the assignment

    Parameters:
    - N_samples: Number of sample frames in sample readings
    - sample_a_markers: LED markers on rigid body A relative to optical tracker
    - sample_b_markers: LED markers on rigid body B relative to optical tracker
    - markers_A: LED markers on rigid body A relative to body
    - markers_B: LED markers on rigid body B relative to body
    - tip_A: Tip of rigid body A relative to body
    
    Returns:
    - Computed sample points
    """

    dk_list = []

    for k in range(N_samples):

        # Extract measured marker positions for this frame
        ai_k = sample_a_markers[k]
        bi_k = sample_b_markers[k]
        
        # Compute FA,k as a Frame object
        FA_k = registration_algo(markers_A, ai_k)
        
        # Compute FB,k as a Frame object
        FB_k = registration_algo(markers_B, bi_k)

        # Transform Atip (tip_A) with respect to rigid body A
        Atip_in_tracker = FA_k.transform_point(tip_A)

        # Inverse FB,k
        FB_k_inv = FB_k.inverse()

        # Compute dk: position of tip_A in with respect to rigid body B
        dk = FB_k_inv.transform_point(Atip_in_tracker)
        dk_list.append(dk)

    return dk_list

def find_closest_point_params(a, p, q, r):
    """
    Algorithm to find parameters lambda and mu of the least squares problem a = lambda(q) + mu(r) + nu(p) where lambda + mu + nu = 1

    Parameters:
    - a: starting points
    - p: vertex of triangle
    - q: vertex of triangle
    - r: vertex of triangle
    
    Returns:
    - lambda: first solution from least squares
    - mu: second solution from least square
    - nu: third solution from least square
    """

    # set up a least squares problem in form Ax = b
    A = np.column_stack((q - p, r - p))
    b = a - p

    # explicitly solve least square by calculating psuedoinverse
    ATA = np.dot(A.T, A)
    ATb = np.dot(A.T, b)
    x = np.dot(np.linalg.inv(ATA), ATb)

    # Calculate nu using the constraint lambda + mu + nu = 1
    nu = 1 - x[0] - x[1]

    return x[0], x[1], nu

def project_on_segment(c, vert1, vert2):
    """
    Algorithm to project a point outside of a triangle onto the triangle's edges

    Parameters:
    - c: point outside the triangle
    - vert1: first vertex of triangle edge
    - vert2: second vertex of triangle edge
    
    Returns:
    - projected point
    """

    # get parameter to use to project the point
    lam = np.dot((c - vert1), (vert2 - vert1)) / np.dot((vert2 - vert1), (vert2 - vert1))
    lam = max(0, min(lam, 1))

    return vert1 + lam * (vert2 - vert1)

def linear_triangle_search(vertices, triangles, points):
    """
    Algorithm to find closest points to an array of given points from triangles

    Parameters:
    - vertices: array of 3D coordinates of vertices to be used to make triangles
    - triangles: array of indicies of vertices that make triangles
    - points: array of given points to find closest points to
    
    Returns:
    - closest_points: array of closest points
    - closest_points_distance: array of euclidean distances between each given point and its closest point
    """

    # set up linear search
    closest_distance = 10000
    closest_point = [0, 0, 0]
    closest_points = []
    closest_points_distance = []

    for point in points:
        for triangle in triangles:

            # define triangle
            p = vertices[triangle[0]]
            q = vertices[triangle[1]]
            r = vertices[triangle[2]]

            # check if closest point exists in triangle
            lam, mu, nu = find_closest_point_params(point, p, q, r)
            c = (lam * q) + (mu * r) + (nu * p)

            # if closest point doesn't exist in triangle, project the point onto the triangle
            if lam < 0:
                c = project_on_segment(c, r, p)
            elif mu < 0:
                c = project_on_segment(c, p, q)
            elif nu < 0:
                c = project_on_segment(c, q, r)

            # update the distance and value of the closest point found if it is better than the last best point
            if np.linalg.norm(c - point) < closest_distance:
                closest_distance = np.linalg.norm(c - point)
                closest_point = c
        
        # create array of closest points and distances
        closest_points.append(closest_point)
        closest_points_distance.append(closest_distance)
        closest_distance = 10000

    return np.array(closest_points), np.array(closest_points_distance)

def bounded_box_search(vertices, triangles, points):
    """
    Algorithm to find closest points to an array of given points from triangles bounded by boxes

    Parameters:
    - vertices: array of 3D coordinates of vertices to be used to make triangles
    - triangles: array of indicies of vertices that make triangles
    - points: array of given points to find closest points to
    
    Returns:
    - closest_points: array of closest points
    - closest_points_distance: array of euclidean distances between each given point and its closest point
    """

    # set up linear search
    closest_distance = 10000
    closest_point = [0, 0, 0]
    closest_points = []
    closest_points_distance = []

    for point in points:
        for triangle in triangles:

            # define triangle
            p = vertices[triangle[0]]
            q = vertices[triangle[1]]
            r = vertices[triangle[2]]

            # create bounding box
            x_min = min(p[0], q[0], r[0])
            x_max = max(p[0], q[0], r[0])

            y_min = min(p[1], q[1], r[1])
            y_max = max(p[1], q[1], r[1])

            z_min = min(p[2], q[2], r[2])
            z_max = max(p[2], q[2], r[2])

            # check if bounded triangle is within the distance of the best point found previously
            if (point[0] >= x_min - closest_distance and point[0] <= x_max + closest_distance) \
                and (point[1] >= y_min - closest_distance and point[1] <= y_max + closest_distance) \
                and (point[2] >= z_min - closest_distance and point[2] <= z_max + closest_distance):

                # check if closest point exists in triangle
                lam, mu, nu = find_closest_point_params(point, p, q, r)
                c = (lam * q) + (mu * r) + (nu * p)

                # if closest point doesn't exist in triangle, project the point onto the triangle
                if lam < 0:
                    c = project_on_segment(c, r, p)
                elif mu < 0:
                    c = project_on_segment(c, p, q)
                elif nu < 0:
                    c = project_on_segment(c, q, r)
            
                 # update the distance and value of the closest point found if it is better than the last best point
                if np.linalg.norm(c - point) < closest_distance:
                    closest_distance = np.linalg.norm(c - point)
                    closest_point = c
            
            # if bounded triangles are outside the distance of the previous point, check the next triangle
            continue
        
        # create array of closest points and distances
        closest_points.append(closest_point)
        closest_points_distance.append(closest_distance)
        closest_distance = 10000

    return np.array(closest_points), np.array(closest_points_distance)


def build_kd_tree(triangles, vertices, depth=0):
    """
    Builds a KD-tree where the pivot is based on the centroid of the triangle.

    Parameters:
    - triangles: Array of triangles (each a list of vertex indices).
    - vertices: Array of 3D vertex coordinates.
    - depth: Current depth of recursion (used to determine splitting axis).

    Returns:
    - KDNode: The root node of the KD-tree.
    """
    n = len(triangles)

    if n <= 0:
        return None

    axis = depth % 3

    # sort triangles according to thier centroids
    sorted_triangles = sorted(
        triangles, 
        key=lambda triangle: ((vertices[triangle[0]] + vertices[triangle[1]] + vertices[triangle[2]]) / 3)[axis]
    )

    # Select the median triangle for splitting.
    median_triangle = sorted_triangles[n // 2]

    # Use the centroid of the median triangle as the pivot.
    pivot = (vertices[median_triangle[0]] + vertices[median_triangle[1]] + vertices[median_triangle[2]]) / 3

    # create subtrees
    left = build_kd_tree(sorted_triangles[:n // 2], vertices, depth + 1)
    right = build_kd_tree(sorted_triangles[(n // 2) + 1:], vertices, depth + 1)

    return KDNode(pivot, median_triangle, left, right)


def kd_tree_search(root, point, vertices, triangles, closest_point, closest_distance, depth=0, lam=0, mu=0, nu=0, closest_triangle=None):
    """
    Searches the KD-tree for the closest point on a triangle to the query point.

    Parameters:
    - root: Current node in the KD-tree.
    - point: Query point.
    - vertices: Array of 3D vertex coordinates.
    - triangles: Array of triangles.
    - closest_point: Closest point found so far.
    - closest_distance: Distance to the closest point found so far.
    - depth: Current depth of recursion (used to determine splitting axis).

    Returns:
    - closest_point: Closest point on a triangle to the query point.
    - closest_distance: Distance to the closest point.
    """
    if root is None:
        return closest_point, closest_distance, lam, mu, nu, closest_triangle

    axis = depth % 3

    # create bounding boxes
    p, q, r = vertices[root.triangle]
    x_min, x_max = min(p[0], q[0], r[0]), max(p[0], q[0], r[0])
    y_min, y_max = min(p[1], q[1], r[1]), max(p[1], q[1], r[1])
    z_min, z_max = min(p[2], q[2], r[2]), max(p[2], q[2], r[2])

    # check if bounded triangle is within the distance of the best point found previously
    if (point[0] >= x_min - closest_distance and point[0] <= x_max + closest_distance) \
        and (point[1] >= y_min - closest_distance and point[1] <= y_max + closest_distance) \
        and (point[2] >= z_min - closest_distance and point[2] <= z_max + closest_distance):

        # Find the closest point on the triangle.
        lam_check, mu_check, nu_check = find_closest_point_params(point, p, q, r)
        c = (lam_check * q) + (mu_check * r) + (nu_check * p)

        if lam_check < 0:
            c = project_on_segment(c, r, p)
        elif mu_check < 0:
            c = project_on_segment(c, p, q)
        elif nu_check < 0:
            c = project_on_segment(c, q, r)

        # check if new point is closer than past point
        dist = np.linalg.norm(c - point)
        if dist < closest_distance:
            closest_distance = dist
            closest_point = c
            lam, mu, nu = lam_check, mu_check, nu_check
            closest_triangle = root.triangle

    # Determine which branch to search first.
    if point[axis] < root.pivot[axis]:
        next_branch = root.left
        opposite_branch = root.right
    else:
        next_branch = root.right
        opposite_branch = root.left

    # Search the closer branch.
    closest_point, closest_distance, lam, mu, nu, closest_triangle = kd_tree_search(next_branch, point, vertices, triangles, closest_point, closest_distance, depth + 1, lam, mu, nu, closest_triangle)

    # account for total triangle space
    epsilon = 5
    # Check if the opposite branch needs to be searched.
    if closest_distance + epsilon >= np.linalg.norm(point[axis] - root.pivot[axis]):
        closest_point, closest_distance, lam, mu, nu, closest_triangle = kd_tree_search(opposite_branch, point, vertices, triangles, closest_point, closest_distance, depth + 1, lam, mu, nu, closest_triangle)

    return closest_point, closest_distance, lam, mu, nu, closest_triangle

def bounded_box_search_kdtree(vertices, triangles, points):
    """
    KD-tree-based closest point search.

    Parameters:
    - vertices: array of vertex coordinates
    - triangles: array of triangle indices
    - points: query points

    Returns:
    - closest_points: array of closest points
    - closest_points_distance: distances to the closest points
    """

    # create a KD tree with triangles and vertices
    root = build_kd_tree(triangles, vertices)
    closest_points = []
    closest_distances = []
    lambdas = []
    mus = []
    nus = []
    closest_triangles = []

    # search tree
    for point in points:
        closest_point, closest_distance, lam, mu, nu, closest_triangle = kd_tree_search(root, point, vertices, triangles, [0, 0, 0], 100000)
        closest_points.append(closest_point)
        closest_distances.append(closest_distance)
        lambdas.append(lam)
        mus.append(mu)
        nus.append(nu)
        closest_triangles.append(closest_triangle)

    return np.array(closest_points), np.array(closest_distances), np.array(lambdas), np.array(mus), np.array(nus), np.array(closest_triangles)


def icp(vertices, triangles, dk_list):
    """
    Implementation of the Iterative Closest Point algorithm

    Parameters:
    - vertices: array of vertex coordinates
    - triangles: array of triangle indices
    - dk_list: tips of rigid body A in rigid body B coordinate frames (inital query points)

    Returns:
    - closest_points: array of closest points
    - closest_points_distance: distances to the closest points
    """

    # set inital F_reg to be an identity
    F_reg = Frame()
    stop_conditions = [0, 0]

    # iterate to find F_reg
    for i in range(100):

        # calculated sample points
        s_k = [F_reg.transform_point(dk) for dk in dk_list]
        s_k = np.hstack(s_k).T

        # find closest points to sample points
        c_k, distances, lambdas, mus, nus, closest_triangles = bounded_box_search_kdtree(vertices, triangles, s_k)

        # get new transformation
        new_F = registration_algo(s_k.T, c_k.T)
        F_reg_check = F_reg.composition(new_F)

        # calculate residual errors between sample points and closest points
        error = s_k - c_k

        # calculate stopping criterion
        stop_condition = np.mean(error**2)
        stop_conditions[1] = stop_conditions[0]
        stop_conditions[0] = stop_condition

        # check if converged
        convergence_bound = 1e-6
        if i > 0 and (abs(stop_conditions[0] - stop_conditions[1]) < convergence_bound or stop_condition < convergence_bound):
            return F_reg, c_k, distances, lambdas, mus, nus, closest_triangles

        # update F_reg
        F_reg = F_reg_check

    return F_reg, c_k, distances, lambdas, mus, nus, closest_triangles
        
def estimate_deformation_find_new_closest(s_k, c_k, lambdas, mus, nus, n_modes, mode_0, modes, closest_triangles, triangles):

    stop_conditions = [0, 0]
    loop_check = 0

    while True:
        all_q0 = []
        q_m = np.zeros((closest_triangles.shape[0], n_modes, 3))
        q_new_stuff = []

        for i, closest_triangle in enumerate(closest_triangles):
            ms_0, mt_0, mu_0 = mode_0[closest_triangle]

            lambda_ = lambdas[i]
            mu = mus[i]
            nu = nus[i]

            q_0 = (lambda_ * ms_0) + (mu * mt_0) + (nu * mu_0)
            all_q0.append(q_0)

            lambda_ = lambdas[i]
            mu = mus[i]
            nu = nus[i]

            q_row = []

            for j in range(n_modes):
                m_s, m_t, m_u = modes[j][closest_triangle]
                q = (lambda_ * m_s) + (mu * m_t) + (nu * m_u)
                q_row.append(q)
                q_m[i, j] = q
            
            q_new_stuff.append(q_row)
        
        all_q0 = np.array(all_q0)
        q_new_stuff  = np.array(q_new_stuff)

        num_triangles = closest_triangles.shape[0]

        # Reshape q_m to (num_triangles * 3, n_modes)
        # A = q_m.reshape(num_triangles * 3, n_modes)
        A = q_new_stuff.reshape(num_triangles * 3, n_modes)

        # Reshape s_k - all_q0 to (num_triangles * 3,)
        b = (s_k - all_q0).reshape(num_triangles * 3)

        # Solve the least squares problem
        mode_weights = np.linalg.lstsq(A, b, rcond=None)[0]

        new_triangle_vertices = mode_0.copy()

        for i in range(mode_0.shape[0]):
            for j in range(n_modes - 1):
                new_triangle_vertices[i] += mode_weights[j] * modes[j][i]

        new_c_k, _, lambdas, mus, nus, closest_triangles = bounded_box_search_kdtree(new_triangle_vertices, triangles, s_k)

        # calculate residual errors between sample points and closest points
        error = c_k - new_c_k

        # calculate stopping criterion
        stop_condition = np.mean(error**2)
        stop_conditions[1] = stop_conditions[0]
        stop_conditions[0] = stop_condition

        if abs(stop_conditions[0] - stop_conditions[1]) < 0.0001:
            break

        c_k = new_c_k
        loop_check += 1

    return new_triangle_vertices, mode_weights

