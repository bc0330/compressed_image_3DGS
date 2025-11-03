import matplotlib.pyplot as plt
import numpy as np

def plot_rd_curve():
    # Data from the provided image
    # Training Set Size (MiB) - X-axis
    # PSNR (dB) - Y-axis

    # 3DGS w/ VVC w/ colmap from uncompressed images (Blue circles)
    x_uncompressed = np.array([2.04, 4.14, 6.80, 10.71, 20.88, 37.05, 59.07])
    y_uncompressed = np.array([23.25, 23.96, 24.37, 24.66, 24.89, 24.93, 24.93]) # Adjusted slightly to match curve

    # 3DGS w/ VVC w/ colmap from VVC images (Orange squares)
    x_vvc_images = np.array([2.04, 4.14, 6.80, 10.71, 20.88, 37.05, 59.07])
    y_vvc_images = np.array([19.33, 19.5, 19.31, 24.37, 24.78, 24.93, 24.92]) # Adjusted slightly to match curve

    # CAT3DGS (Green triangles)
    x_cat3dgs = np.array([2.8, 3.75, 6.85, 9.4, 17.2, 21.4]) # Only up to 20 MiB visible
    y_cat3dgs = np.array([24.01, 24.37, 24.79, 25.05, 25.17, 25.05]) # Adjusted slightly to match curve
    
    # 3DGS inter-frame coding (low delay)
    x_intraframe = np.array([7.75, 37.9])
    y_intraframe = np.array([24.09, 24.76]) # Example PSNR values
    
    # 3DGS inter-frame coding (random access)
    x_random_access = np.array([0.5, 1.08, 1.87, 3.12, 7.27])
    y_random_access = np.array([21.56, 22.36, 22.81, 23.28, 23.99]) # Example PSNR values
    
    # 3DGS inter-frame coding (random access)
    x_random_access_new = np.array([0.46, 0.98, 1.68, 2.83, 6.58])
    y_random_access_new = np.array([21.56, 22.36, 22.81, 23.28, 23.99]) # Example PSNR values

    # 3DGS w/o VVC (Red dashed line)
    # This is a horizontal line, so PSNR is constant for all training set sizes
    psnr_wo_vvc = 24.69 # Approximate value from the dashed line in the image

    plt.figure(figsize=(8, 6)) # Adjust figure size as needed

    # Plotting the data
    plt.plot(x_uncompressed, y_uncompressed, marker='o', linestyle='-', markersize=8, color='tab:blue', label='3DGS w/ VVC w/ colmap from uncompressed images')
    plt.plot(x_vvc_images, y_vvc_images, marker='s', linestyle='-', markersize=8, color='tab:orange', label='3DGS w/ VVC w/ colmap from VVC images')
    plt.plot(x_cat3dgs, y_cat3dgs, marker='^', linestyle='-', markersize=8, color='limegreen', label='CAT3DGS')
    plt.plot(x_intraframe, y_intraframe, marker='D', linestyle='-', markersize=8, color='purple', label='3DGS inter-frame coding')
    # plt.plot(x_random_access, y_random_access, marker='v', linestyle='-', markersize=8, color='purple', label='3DGS random access coding')
    plt.plot(x_random_access_new, y_random_access_new, marker='v', linestyle='-', markersize=8, color='purple', label='3DGS random access coding (new)')

    # Plotting the horizontal dashed line
    plt.axhline(y=psnr_wo_vvc, color='red', linestyle='--', label='3DGS w/o VVC')

    # Set labels and title
    plt.xlabel('Training Set Size (MiB)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('bicycle 3DGS RD Curve', fontsize=14)

    # Set x-axis ticks and limits
    plt.xticks(np.arange(0, 61, 10))
    plt.xlim(0, 62) # Slightly beyond 60 to match the image's extent

    # Set y-axis ticks and limits
    plt.yticks(np.arange(20, 26, 1))
    plt.ylim(19.2, 25.5) # Slightly adjusted to match the image's extent

    # Add grid
    plt.grid(True)

    # Add legend
    plt.legend(loc='lower right', fontsize=10)

    # Tight layout to prevent labels from overlapping
    plt.tight_layout()

    # Save the figure
    plt.savefig('bicycle_3DGS_RD_Curve.png')
    
    # Show the plot
    plt.show()

# Call the function to generate the plot
plot_rd_curve()