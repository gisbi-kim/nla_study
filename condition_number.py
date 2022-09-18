import numpy as np

print("==================================================")
print("A study on the Condition number of a linear system")
print("==================================================")

print("\n=====A motivation example=====")
# we will see this system matrix is ill-conditioned (i.e., cond(A) >> 1)
# which means very sensitive to the data noise
A = np.array([8, 6, 4, 1, 1, 4, 5, 1, 8, 4, 1, 1, 1, 4, 3, 6]).reshape(4, 4)
print(f"A is\n {A}")

# motivation: bad solution example
# see p76 of 2008 book Numerical Linear Algebra (Grégoire Allaire and Sidi Mahmoud Kaber)
b_true = np.array([19, 11, 14, 14])
x_true = np.linalg.solve(A, b_true)
print(f"x_true for b_true {b_true}\n is {x_true}")

b_noised = np.array([19.01, 11.05, 14.07, 14.05])
x_noised = np.linalg.solve(A, b_noised)
print(f"x for the very slight noised data b_noise {b_noised}\n is {x_noised}")
print(f" The solution has been gone to far apart from the original expected one")

"""
 The definition of the condition number
"""
print("\n=====The definition of the condition number=====")
# the condition number tells us the sensitivity of a solution w.r.t the data or model noise.
# ref: see p80 of 2008 book Numerical Linear Algebra (Grégoire Allaire and Sidi Mahmoud Kaber)
A_inv = np.linalg.inv(A)
print(f"A_inv is\n {A_inv}")

A_norm = np.linalg.norm(A, 2)
print(f"||A|| is {A_norm:.2f}")

A_inv_norm = np.linalg.norm(A_inv, 2)
print(f"||A_inv|| is {A_inv_norm:.2f}")

cond_A = A_norm * A_inv_norm
print(f"cond(A) = ||A||2 * ||A^(-1)||2 is {cond_A:.2f}")


"""
 A method 2 to get the condition number
"""
print("\n=====A method 2 to get the condition number=====")
# For 2-norm, the cond_A = s1(A) / sn(A), where sn and s1 are the smallest and the largest singular values of A
# ref: see p82 of 2008 book Numerical Linear Algebra (Grégoire Allaire and Sidi Mahmoud Kaber)
svd_result = np.linalg.svd(A)
u, s, vh = svd_result
print(s)
cond2_A = s[0] / s[-1]
print(f"cond2(A) = s1/sn is {cond2_A:.2f}")


"""
 A method 3 to get the condition number
"""
print("\n=====A method 3 to get the condition number=====")
# The condition number is invariant under an orthogonal transformation
# thus, cond(A) == cond(R), where A = QR
# ref: see 119 of 2008 book Numerical Linear Algebra (Grégoire Allaire and Sidi Mahmoud Kaber)
# ref2: see p114 of 1996 book SIAM a Numerical Methods for Least Squares Problems
Q, R = np.linalg.qr(A)
print(f"Q:\n{Q}\n R:\n{R}")

R_norm = np.linalg.norm(R, 2)
R_inv = np.linalg.inv(R)
R_inv_norm = np.linalg.norm(R_inv, 2)
print(f"R norm: {R_norm:.2f}, R_inv norm: {R_inv_norm:.2f}")

cond_R = R_norm * R_inv_norm
print(f"cond2(R) = ||R||2 * ||R^(-1)||2 is {cond_R:.2f}")
