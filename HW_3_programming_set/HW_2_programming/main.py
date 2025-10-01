import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl


def image_loader(args):
    """
    Output:
        I: the image matrix of size N-by-M-by-3.
            The first dimension is horizontal, left-right (N pixels).
            The second dimension is vertical, bottom-up (M pixels).
            The third dimension is color channels, from Red to Green to Blue.
            The Red, Green, and Blue values are within [0, 1].
    """

    if args.data in ["helmet", "stop"]:
        print("Using " + args.data + " photo")
        current_dir = os.getcwd()
        image_path = osp.join(current_dir, 'data', args.data + '.png')
        I = np.asarray(Image.open(image_path)).astype(np.float64)/255
        I = np.transpose(I, (1, 0, 2))
        I = I[:,::-1,:]

    ## Display the input image
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(I, (1, 0, 2)), origin='lower')
    ax.set_title('Inpur image: I')
    if args.display:
        plt.show()
    plt.close(fig)

    return I


def point_cloud_loader(args):
    """
    Output:
        P: the 3D point list of size K-by-6.
            K: number of points
            6: the first 3 dimensions are "X, Y, and Z" locations; the final 3 dimensions are "Red, Green, and Blue" color channels
            The Red, Green, and Blue values are within [0, 1].
    """

    if args.data in ["helmet", "stop"]:
        print("Using " + args.data + " photo")
        current_dir = os.getcwd()
        image_path = osp.join(current_dir, 'data', args.data + '.png')
        I = np.asarray(Image.open(image_path)).astype(np.float64)/255
        I = np.transpose(I, (1, 0, 2))
        I = I[:,::-1,:]
        I_shape = I.shape
        N = I_shape[0]
        M = I_shape[1]
        K = N * M
        X = np.matmul((np.linspace(-N/2.0, N/2.0, num=N)).reshape((N, 1)), np.ones((1, M)))
        Y = np.matmul(np.ones((N, 1)), (np.linspace(-M/2.0, M/2.0, num=M)).reshape((1, M)))
        Z = ones((N, M))
        P = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), I[:,:,0].reshape(-1, 1), I[:,:,1].reshape(-1, 1), I[:,:,2].reshape(-1, 1)), axis = 1)
        
    return P


def array_to_list(args):
    """
    Input:
        I: input image, a 3D matrix of size N-by-M-by-3
    Output:
        I_out: the list of pixels of size K-by-5.
            K: number of pixels
            5: the first 3 dimensions are "X, Y" locations; the final 3 dimensions are "Red, Green, and Blue" color channels
            The Red, Green, and Blue values are within [0, 1].
    """

    I_shape = I.shape
    N = I_shape[0]
    M = I_shape[1]
    K = N * M
    X = np.matmul((np.linspace(0, N - 1, num=N)).reshape((N, 1)), np.ones((1, M)))
    Y = np.matmul(np.ones((N, 1)), (np.linspace(0, M - 1, num=M)).reshape((1, M)))
    P = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), , I[:, :, 0].reshape(-1, 1), I[:, :, 1].reshape(-1, 1), I[:, :, 2].reshape(-1, 1)), axis=1)

    ## Save the points
    if args.save:
        current_dir = os.getcwd()
        np.savez(osp.join(current_dir, 'result', args.data + '_' + 'pixel_list.npz'), m = P)

    return P


def translation_matrix(tx, ty):
    """
    Input:
        tx: horizontal offset
        ty: vertical offset
    Output:
        mat_out: a 3-by-3 translation matrix
    TODO:
        You are to create a 3-by-3 translation matrix
    """

    #### Your job 1.1 starts here: translation ####


    #### Your job 1.1 ends here: translation ####

    return mat_out


def shearing_matrix(sx, sy):
    """
    Input:
        sx: scaling along x
        sy: scaling along y
    Output:
        mat_out: a 3-by-3 translation matrix
    TODO:
        You are to create a 3-by-3 scaling matrix
    """

    #### Your job 1.2 starts here: translation ####


    #### Your job 1.2 ends here: translation ####

    return mat_out


def rotation_matrix(theta):
    """
    Input:
        theta: rotation degree counterclockwise
    Output:
        mat_out: a 3-by-3 translation matrix
    TODO:
        You are to create a 3-by-3 rotation matrix
    """

    #### Your job 1.2 starts here: translation ####


    #### Your job 1.2 ends here: translation ####

    return mat_out


def shearing_matrix(qx, qy):
    """
    Input:
        qx: shearing qx
        qy: shearing qy
    Output:
        mat_out: a 3-by-3 translation matrix
    TODO:
        You are to create a 3-by-3 shearing matrix
    """

    #### Your job 1.4 starts here: translation ####


    #### Your job 1.4 ends here: translation ####

    return mat_out


def image_geometry_transformation(args):

    """
    Input:
        I: a 3D matrix of size N-by-M-by-3
    Output:
        I_out_amplitude: DFT amplitude, a 3D matrix of size N-by-M-by-3. The third dimension is for Red-Green-Blue.
        I_out_phase: DFT phase, a 3D matrix of size N-by-M-by-3. The third dimension is for Red-Green-Blue.
    TODO:
        You are to perform Discrete Fourier Transform. See the Homework slide deck for details.
    """

    ## Initiate the output
    if np.size(I.shape) != 3:
        I = I.reshape((I.shape[0], I.shape[1], 1))
    I_out_real = np.zeros(I.shape)
    I_out_imaginary = np.zeros(I.shape)
    I_out_amplitude = np.zeros(I.shape)
    I_out_phase = np.zeros(I.shape)
    N = I.shape[0]
    M = I.shape[1]

    #### Your job 2 starts here: DFT (you may add just 5~10 more line without for loops; the fewer foor loops you have; the faster the code runs) ####

    for c in range(I.shape[2]):
        for u in range(I.shape[0]):
            for v in range(I.shape[1]):

                # Please fill in I_out_real and I_out_imaginary

    #### Your job 2 ends here: DFT ####

    ## Calculate amplitude and phase
    I_out_real[abs(I_out_real) < 10 ** -11] = 0
    I_out_imaginary[abs(I_out_imaginary) < 10 ** -11] = 0
    I_out_amplitude = np.absolute(I_out_real + 1j * I_out_imaginary)
    I_out_phase = np.angle(I_out_real + 1j * I_out_imaginary)

    ## Display the amplitude: for the Red channel
    temp_I = display_origin_in_center(args, I_out_amplitude)
    fig, ax = plt.subplots()
    ax.imshow((temp_I[:,:,0].transpose()) ** 0.25, origin='lower', cmap='gray', vmin=0, vmax=np.max(temp_I ** 0.25), extent=[-int((I.shape[0] + 1) / 2), int((I.shape[0] + 1) / 2) - 1, -int((I.shape[1] + 1) / 2), int((I.shape[1] + 1) / 2) - 1])
    ax.set_title('DFT_amplitude')
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', str(args.current_step) + '_' + 'DFT_amplitude_' + args.data + '.png'))
        np.savez(osp.join(current_dir, 'result', str(args.current_step) + '_' +  'Results_DFT_amplitude_' + args.data + '.npz'), m = temp_I)
    if args.display:
        plt.show()
    plt.close(fig)

    ## Display the phase: for the Red channel
    temp_I = display_origin_in_center(args, I_out_phase)
    fig, ax = plt.subplots()
    ax.imshow(temp_I[:,:,0].transpose(), origin='lower', cmap='gray', vmin = - np.pi, vmax = np.pi, extent=[-int((I.shape[0] + 1) / 2), int((I.shape[0] + 1) / 2) - 1, -int((I.shape[1] + 1) / 2), int((I.shape[1] + 1) / 2) - 1])
    ax.set_title('DFT_phase')
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', str(args.current_step) + '_' + 'DFT_Phase_' + args.data + '.png'))
        np.savez(osp.join(current_dir, 'result', str(args.current_step) + '_' + 'Results_DFT_Phase_' + args.data + '.npz'), m = temp_I)
    if args.display:
        plt.show()
    plt.close(fig)

    return I_out_amplitude, I_out_phase


def IDFT(args, I_amplitude, I_phase):
    """
    Input:
        I_amplitude: DFT amplitude, a 3D matrix of size N-by-M-by-3. The third dimension is for Red-Green-Blue.
        I_phase: DFT phase, a 3D matrix of size N-by-M-by-3. The third dimension is for Red-Green-Blue.
    Output:
        I_out: reconstructed image, a 3D matrix of size N-by-M-by-3
    TODO:
        You are to perform inverse Discrete Fourier Transform. See the Homework slide deck for details.
    """

    ## Initiate the output
    if np.size(I_amplitude.shape) != 3:
        I_amplitude = I_amplitude.reshape((I_amplitude.shape[0], I_amplitude.shape[1], 1))
    if np.size(I_phase.shape) != 3:
        I_phase = I_phase.reshape((I_phase.shape[0], I_phase.shape[1], 1))
    I_out = np.zeros(I_amplitude.shape)
    N = I_amplitude.shape[0]
    M = I_amplitude.shape[1]

    ## IDFT (inverse DFT)
    matA = (1 / N) * np.matmul(np.array(range(I_out.shape[0])).reshape((-1, 1)), np.ones((1, I_out.shape[1])))
    matB = (1 / M) * np.matmul(np.ones((I_out.shape[0], 1)), np.array(range(I_out.shape[1])).reshape((1, -1)))

    for c in range(I_out.shape[2]):
        for n in range(I_out.shape[0]):
            for m in range(I_out.shape[1]):
                matAB = n * matA + m * matB
                I_out[n, m, c] = (np.sum(I_amplitude[:, :, c] * np.cos(I_phase[:, :, c]) * np.cos(2 * np.pi * matAB)) \
                                 - np.sum(I_amplitude[:, :, c] * np.sin(I_phase[:, :, c]) * np.sin(2 * np.pi * matAB))) /N/M

    ## Display the IDFT result
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(I_out, (1, 0, 2)), origin='lower')
    ax.set_title('IDFT_output')
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(osp.join(current_dir, 'result', str(args.current_step) + '_' + 'IDFT_image_' + args.data + '.png'))
        np.savez(osp.join(current_dir, 'result', str(args.current_step) + '_' + 'Results_IDFT_image_' + args.data + '.npz'), m = I_out)
    if args.display:
        plt.show()
    plt.close(fig)

    return I_out


def Periodic_basis(args, u_freq, v_freq, phase):
    """
    Input:
        u: modulation horizontal frequency
        v: modulation vertical frequency
        phase
    Output:
        I: output image, a 3D matrix of size N-by-M-by-3
    """

    ## This function generates cosine images
    matA = (1 / 31) * np.matmul(np.array(range(31)).reshape((-1, 1)), np.ones((1, 31)))
    matB = (1 / 31) * np.matmul(np.ones((31, 1)), np.array(range(31)).reshape((1, -1)))
    matAB = u_freq * matA + v_freq * matB
    I = np.cos(2 * np.pi * matAB + phase).reshape((31, 31, 1))
    I = np.concatenate((I, I, I), axis = 2)

    return I


def Recover_function(args, I, u_target_freq, v_target_freq):
    """
    Input:
        I: the mixed image
        u_target_freq: the target horizontal frequency
        v_target_freq: the target vertical frequency
    Output:
        I_out: the recovered target image
    TODO:
        Implement the recovering function of the target frequency image from I
    """

    ############### Caution ###########################################################################
    ## You cannot simply use u_target_freq, v_target_freq, and Periodic_basis to regenerate the target image
    ##
    ## You can reuse any function in this main.py file
    ## For example, you may create your own kernel---based on u_target_freq and v_target_freq---and use convolution to recover I_target
    ##
    ## You can even directly do some manipulations in the frequency domain and reconstruct the target image
    ## For example, you may perform DFT(args, I), edit the frequency responses, and perform IDFT to recover the target image
    ##
    ## Your solution cannot exceed 20 lines
    ############### Caution ###########################################################################

    #### Your implementation starts here (your solution cannot exceed 20 lines) ####
    I_out = np.zeros(I.shape) # this is a placeholder, and you can remove this line completely
    
    #### Your implementation ends here ####

    print("Display the recovered target image")
    fig, ax = plt.subplots()
    ax.imshow(np.transpose(I_out, (1, 0, 2)), origin='lower')
    ax.set_title('Recovered image: I_out')
    if args.save:
        current_dir = os.getcwd()
        plt.savefig(
            osp.join(current_dir, 'result', str(args.current_step) + '_' + 'Recovered_image_' + args.data + '.png'))
        np.savez(osp.join(current_dir, 'result',
                          str(args.current_step) + '_' + 'Results_Recovered_image_' + args.data + '.npz'), m = I_out)
    plt.show()
    plt.close(fig)

    return I_out


def main(args):

    ## Convolution
    if int(args.current_step) == 1:
        print("Load image")
        I = image_loader(args)
        print("Perfrom convolution")
        kernel = load_kernel(args)
        I_out = Convolution(args, I, kernel)


    ## DFT
    if int(args.current_step) == 2:
        print("Load image")
        I = image_loader(args)
        print("Perfrom DFT")
        I_amplitude, I_phase = DFT(args, I)
        print("Perfrom IDFT")
        I_out = IDFT(args, I_amplitude, I_phase)

        if np.sum((I - I_out) ** 2) > (10 ** (-15)): # Check if your DFT is implemented correctly
            print("Wrong implementation in DFT")


    ## Modulation
    if int(args.current_step) == 3:
        print("Load image")
        I = image_loader(args)
        print("Perform modulation")
        u_freq = 5
        v_freq = 5
        print("Show modulated image")
        I_mod_1 = Modulation(args, I, u_freq, v_freq)
        print("Show modulated frequency responses")
        I_mod_amplitude, I_mod_phase = DFT(args, I_mod_1)

        print("Perform convolution in frequency domain to mimic modulation")
        kernel = np.zeros((11, 11))
        kernel[0, 0] = 0.5
        kernel[10, 10] = 0.5
        I_amplitude, I_phase = DFT(args, I)
        I_real = I_amplitude * np.cos(I_phase)
        I_imaginary =I_amplitude * np.sin(I_phase)
        print("Show convolution in frequency responses")
        I_conv_real = Convolution(args, display_origin_in_center(args, I_real), kernel)
        I_conv_imaginary = Convolution(args, display_origin_in_center(args, I_imaginary), kernel)
        print("Show reconstructed image")
        I_conv_real[abs(I_conv_real) < 10 ** -12] = 0
        I_conv_imaginary[abs(I_conv_imaginary) < 10 ** -12] = 0
        I_conv_amplitude = np.absolute(I_conv_real + 1j * I_conv_imaginary)
        I_conv_phase = np.angle(I_conv_real + 1j * I_conv_imaginary)
        I_mod_2 = IDFT(args, display_origin_in_corner(args, I_conv_amplitude), display_origin_in_corner(args, I_conv_phase))


    ## Reconstruct the translation kernel
    if int(args.current_step) == 4:
        print("Load image")
        I = image_loader(args)
        print("Create the translated image")
        kernel = load_kernel(args)
        I_out = Convolution(args, I, kernel)
        print("DFT of the input image")
        I_amplitude, I_phase = DFT(args, I)
        print("DFT of the output image")
        I_out_amplitude, I_out_phase = DFT(args, I_out)
        I_div_amplitude = I_out_amplitude / (I_amplitude + (10 ** -12))
        I_div_phase = I_out_phase - I_phase
        kernel_reconstruct = IDFT(args, I_div_amplitude, I_div_phase)
        print("Reconstruct the convolution kernel")
        kernel_reconstruct = display_origin_in_center(args, kernel_reconstruct)

        ## Display the reconstructed kernel
        fig, ax = plt.subplots()
        ax.imshow(kernel_reconstruct[:, :, 0].transpose(), origin='lower', cmap='gray',
                  extent=[-int((kernel_reconstruct.shape[0] + 1) / 2), int((kernel_reconstruct.shape[0] + 1) / 2) - 1,
                          -int((kernel_reconstruct.shape[1] + 1) / 2), int((kernel_reconstruct.shape[1] + 1) / 2) - 1])
        ax.set_title('Reconstructed kernel')
        plt.show()
        plt.close(fig)


    ## Mixing phase
    if int(args.current_step) == 5:
        print("Load image")
        I = image_loader(args)
        print("Mixing phases")
        print("Load the second image")
        args.data = 'helmet'
        I_2 = image_loader(args)
        I_amplitude, I_phase = DFT(args, I)
        I_amplitude_2, I_phase_2 = DFT(args, I_2)
        _ = IDFT(args, I_amplitude_2, I_phase) # Image 1 phase + Image 2 amplitude
        _ = IDFT(args, I_amplitude, I_phase_2) # Image 2 phase + Image 1 amplitude


    ## Denoising
    if int(args.current_step) == 6:
        print("Load image")
        I = image_loader(args)
        print("Create the noisy image")
        I_noisy = I + 0.2 * np.random.rand(I.size).reshape(I.shape)

        ## Display the noisy image
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(I_noisy, (1, 0, 2)), origin='lower')
        ax.set_title('Noisy image: I_noisy')
        plt.show()
        plt.close(fig)

        print("Perfrom denoising")
        kernel = load_kernel(args)
        _ = Convolution(args, I_noisy, kernel)


    ## Detect edges
    if int(args.current_step) == 7:
        print("Load image")
        I = image_loader(args)
        print("Perfrom image Laplacian")
        kernel = load_kernel(args)
        _ = Convolution(args, I, kernel)


    ## Recover the target image
    if int(args.current_step) == 8:
        print("Create the target image")
        u_target_freq = 3
        v_target_freq = 3
        I_target = Periodic_basis(args, u_target_freq, v_target_freq, 0)
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(I_target, (1, 0, 2)), origin='lower')
        ax.set_title('Target image: I_target')
        plt.show()
        plt.close(fig)

        print("Create the mixed image")
        I_1 = Periodic_basis(args, -8, 14, 0.31 * np.pi)
        I_2 = Periodic_basis(args, 10, 9, -0.44 * np.pi)
        I_3 = Periodic_basis(args, 7, -12, 0.78 * np.pi)
        I_4 = Periodic_basis(args, 8, -6, 1 * np.pi)
        I = I_target + I_1 + I_2 + I_3 + I_4

        print("Display the mixed image")
        fig, ax = plt.subplots()
        ax.imshow(np.transpose(I, (1, 0, 2))/8 + 0.5, origin='lower')
        ax.set_title('Mixed image: I')
        plt.show()
        plt.close(fig)

        #### Your job 4 starts here: implement the Recover_function ####
        I_out = Recover_function(args, I, u_target_freq, v_target_freq)
        #### Your job 4 ends here: implement the Recover_function ####

        if np.sum((I_target - I_out) ** 2) > (10 ** (-12)): # Check if your recovering is good enough
            print("You can do better!")
        else:
            print("Your result looks good!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convolution_and_Fourier_and_Filter")
    parser.add_argument('--path', default="data", type=str)
    parser.add_argument('--data', default="none", type=str)
    parser.add_argument('--kernel', default="none", type=str)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--current_step', default=0, type=int)
    args = parser.parse_args()
    main(args)

    # Fill in the other students you collaborate with:
    # e.g., Wei-Lun Chao, chao.209
    #
    #