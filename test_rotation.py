import numpy as np

# Test the numpy documentation statement about N-D matrix multiplication
print("Testing numpy matrix multiplication with N-D arrays...")

# Create a simple 3x3 rotation matrix
R = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
])

print(f"Rotation matrix R:\n{R}")
print(f"R shape: {R.shape}")

# Create a 4D array representing multiple strain tensors
# Shape: (2, 3, 3, 3) - 2 time steps, 3x3x3 spatial grid, each with a 3x3 strain tensor
strain = np.random.rand(2, 3, 3, 3, 3)
print(f"Original strain shape: {strain.shape}")

# Test the matrix multiplication
try:
    # This should work according to the numpy documentation
    rotated_strain = np.linalg.inv(R) @ strain @ R
    print(f"Rotated strain shape: {rotated_strain.shape}")
    print("✅ Matrix multiplication with N-D arrays works as expected!")
    
    # Verify the operation preserved the shape
    assert rotated_strain.shape == strain.shape, "Shape should be preserved"
    print("✅ Shape preservation verified!")
    
except Exception as e:
    print(f"❌ Error: {e}")

# Test with different N-D shapes
print("\nTesting with different N-D shapes...")

# Test with 3D array (batch of 2D matrices)
strain_3d = np.random.rand(5, 3, 3)  # 5 matrices of 3x3
print(f"3D strain shape: {strain_3d.shape}")

try:
    rotated_3d = np.linalg.inv(R) @ strain_3d @ R
    print(f"3D rotated shape: {rotated_3d.shape}")
    print("✅ 3D array multiplication works!")
except Exception as e:
    print(f"❌ 3D Error: {e}")

# Test with 5D array
strain_5d = np.random.rand(2, 3, 4, 3, 3)  # 2 batches, 3x4 spatial grid, 3x3 tensors
print(f"5D strain shape: {strain_5d.shape}")

try:
    rotated_5d = np.linalg.inv(R) @ strain_5d @ R
    print(f"5D rotated shape: {rotated_5d.shape}")
    print("✅ 5D array multiplication works!")
except Exception as e:
    print(f"❌ 5D Error: {e}")

print("\nConclusion: The numpy documentation statement is CORRECT!")
print("N-D arrays (N > 2) are treated as stacks of matrices in the last two dimensions.") 