import numpy as np
import matplotlib.pyplot as plt

from constraintSystem import constraintSystem

datadir = './Plots/'
fontsize = 18
legendFontSize=10

# Set global tick size for both x and y axes
plt.rcParams['xtick.labelsize'] = 12  # Set the size for x-axis ticks
plt.rcParams['ytick.labelsize'] = 12  # Set the size for y-axis ticks

makedetplot = True
makestochplot = True

if makedetplot :
    print("Generic linear system with small stochastic noise")

    print('Creating the correlation matrix')
    dphi_x = 0.5
    dphi_y = 0.3
    dphi_z = 0.4
    alpha_x = 1-1.0e-6
    alpha_y = 1-1.0e-6
    alpha_z = 1-1.0e-6
    g=0.3  #0.1 is a good choice
    Q = np.array([
        [alpha_x*np.cos(dphi_x), -alpha_x*np.sin(dphi_x), 0, 0, g, 0],
        [alpha_x*np.sin(dphi_x),  alpha_x*np.cos(dphi_x), 0, 0, 0, g],
        [0, 0, alpha_y*np.cos(dphi_y), -alpha_y*np.sin(dphi_y), g, 0],
        [0, 0, alpha_y*np.sin(dphi_y),  alpha_y*np.cos(dphi_y), 0, g],
        [0, 0, 0, 0, alpha_z*np.cos(dphi_z), -alpha_z*np.sin(dphi_z)],
        [0, 0, 0, 0, alpha_z*np.sin(dphi_z),  alpha_z*np.cos(dphi_z)]
        ])

    sigma_x = 0.0005
    sigma_y = 0.0005
    sigma_z = 0.0005
    S = np.array([
        [sigma_x, 0, 0, 0, 0, 0],
        [0, sigma_x, 0, 0, 0, 0],
        [0, 0, sigma_y, 0, 0, 0],
        [0, 0, 0, sigma_y, 0, 0],
        [0, 0, 0, 0, sigma_z, 0],
        [0, 0, 0, 0, 0, sigma_z]
        ])

    Ncc=10
    XX = constraintSystem(Q, S, kmax=Ncc,Niter=1000000)
    print()
    print(f"The diagonal part of the corelation matrix: \n {np.array2string(np.diagonal(XX.C), formatter={'float': lambda x: f'{x:.2f}'}, separator=', ')}")
    sigma21 = XX.C[0,0]
    sigme2d = XX.C[4,4]

    #################################################################x

    print('constrain subsystem 1')
    allc1 = [ (i,0) for i in range(1,Ncc)]
    allc2 = [ (i,1) for i in range(1,Ncc)]

    Naccur = 15
    accur = [ np.exp(u) for u in np.linspace( -6*np.log(10), 1*np.log(10), Naccur) ]
    data = np.zeros(shape= (Naccur, Ncc, *Q.shape))
    for a,acc in enumerate(accur):
        for i in range(Ncc):
            data[a, i] = XX.system(allc1[:i] + allc2[:i], acc)[0]

    filename = 'plot_det_c1_v1_Noc.pdf'
    print(f'   variance of subsystem 1 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 0,0]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(r'variance of the $x$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c1_v1_bin.pdf'
    print(f'   variance of subsystem 1 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 0,0]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c1_v2_Noc.pdf'
    print(f'   variance of subsystem 2 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 2,2]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c1_v2_bin.pdf'
    print(f'   variance of subsystem 2 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 2,2]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c1_v3_Noc.pdf'
    print(f'   variance of subsystem 3 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 4,4]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c1_v3_bin.pdf'
    print(f'   variance of subsystem 3 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 4,4]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    #################################################################x

    print('constrain subsystem 3')
    allc1 = [ (i,4) for i in range(1,Ncc)]
    allc2 = [ (i,5) for i in range(1,Ncc)]

    Naccur = 15
    accur = [ np.exp(u) for u in np.linspace( -6*np.log(10), 1*np.log(10), Naccur) ]
    data = np.zeros(shape= (Naccur, Ncc, *Q.shape))
    for a,acc in enumerate(accur):
        for i in range(Ncc):
            data[a, i] = XX.system(allc1[:i] + allc2[:i], acc)[0]

    filename = 'plot_det_c3_v1_Noc.pdf'
    print(f'   variance of subsystem 1 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 0,0]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c3_v1_bin.pdf'
    print(f'   variance of subsystem 1 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 0,0]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c3_v2_Noc.pdf'
    print(f'   variance of subsystem 2 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 2,2]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c3_v2_bin.pdf'
    print(f'   variance of subsystem 2 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 2,2]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c3_v3_Noc.pdf'
    print(f'   variance of subsystem 3 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 4,4]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c3_v3_bin.pdf'
    print(f'   variance of subsystem 3 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 4,4]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    #################################################################x

    print('constrain subsystem 1 and 2')
    allc1 = [ (i,0) for i in range(1,Ncc)]
    allc2 = [ (i,1) for i in range(1,Ncc)]
    allc3 = [ (i,2) for i in range(1,Ncc)]
    allc4 = [ (i,3) for i in range(1,Ncc)]

    Naccur = 15
    accur = [ np.exp(u) for u in np.linspace( -6*np.log(10), 1*np.log(10), Naccur) ]
    data = np.zeros(shape= (Naccur, Ncc, *Q.shape))
    for a,acc in enumerate(accur):
        data[a, 0] = XX.system([], acc)[0]
        for i in range(Ncc-1):
            data[a, i+1] = XX.system(allc1[:i] + allc2[:i] + allc3[:1] + allc4[:1], acc)[0]

    filename = 'plot_det_c12_v1_Noc.pdf'
    print(f'   variance of subsystem 1 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 0,0]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c12_v1_bin.pdf'
    print(f'   variance of subsystem 1 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 0,0]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c12_v2_Noc.pdf'
    print(f'   variance of subsystem 2 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 2,2]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c12_v2_bin.pdf'
    print(f'   variance of subsystem 2 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 2,2]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c12_v3_Noc.pdf'
    print(f'   variance of subsystem 3 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 4,4]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c12_v3_bin.pdf'
    print(f'   variance of subsystem 3 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 4,4]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    #################################################################x

    print('constrain subsystem 1 and 3')
    allc1 = [ (i,0) for i in range(1,Ncc)]
    allc2 = [ (i,1) for i in range(1,Ncc)]
    allc3 = [ (i,4) for i in range(1,Ncc)]
    allc4 = [ (i,5) for i in range(1,Ncc)]

    Naccur = 15
    accur = [ np.exp(u) for u in np.linspace( -6*np.log(10), 1*np.log(10), Naccur) ]
    data = np.zeros(shape= (Naccur, Ncc, *Q.shape))
    for a,acc in enumerate(accur):
        for i in range(Ncc):
            data[a, i] = XX.system(allc1[:i] + allc2[:i] + allc3[:i] + allc4[:i], acc)[0]

    filename = 'plot_det_c13_v1_Noc.pdf'
    print(f'   variance of subsystem 1 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 0,0]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c13_v1_bin.pdf'
    print(f'   variance of subsystem 1 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 0,0]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c13_v2_Noc.pdf'
    print(f'   variance of subsystem 2 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 2,2]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c13_v2_bin.pdf'
    print(f'   variance of subsystem 2 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 2,2]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c13_v3_Noc.pdf'
    print(f'   variance of subsystem 3 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 4,4]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_det_c13_v3_bin.pdf'
    print(f'   variance of subsystem 3 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 4,4]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)


#########################################################xx
#######################################################xxxx

if makestochplot:
    print("Generic linear system with small stochastic noise")

    print('Creating the correlation matrix')
    dphi_x = 0.5
    dphi_y = 0.3
    dphi_z = 0.4
    alpha_x = 1-1.0e-4
    alpha_y = 1-1.0e-4
    alpha_z = 1-1.0e-4
    g=0.1  #0.1 is a good choice
    Q = np.array([
        [alpha_x*np.cos(dphi_x), -alpha_x*np.sin(dphi_x), 0, 0, g, 0],
        [alpha_x*np.sin(dphi_x),  alpha_x*np.cos(dphi_x), 0, 0, 0, g],
        [0, 0, alpha_y*np.cos(dphi_y), -alpha_y*np.sin(dphi_y), g, 0],
        [0, 0, alpha_y*np.sin(dphi_y),  alpha_y*np.cos(dphi_y), 0, g],
        [0, 0, 0, 0, alpha_z*np.cos(dphi_z), -alpha_z*np.sin(dphi_z)],
        [0, 0, 0, 0, alpha_z*np.sin(dphi_z),  alpha_z*np.cos(dphi_z)]
        ])

    sigma_x = 0.010
    sigma_y = 0.010
    sigma_z = 0.010
    S = np.array([
        [sigma_x, 0, 0, 0, 0, 0],
        [0, sigma_x, 0, 0, 0, 0],
        [0, 0, sigma_y, 0, 0, 0],
        [0, 0, 0, sigma_y, 0, 0],
        [0, 0, 0, 0, sigma_z, 0],
        [0, 0, 0, 0, 0, sigma_z]
        ])

    Ncc=10
    XX = constraintSystem(Q, S, kmax=Ncc,Niter=1000000)
    print()
    print(f"The diagonal part of the corelation matrix: \n {np.array2string(np.diagonal(XX.C), formatter={'float': lambda x: f'{x:.2f}'}, separator=', ')}")
    sigma21 = XX.C[0,0]
    sigme2d = XX.C[4,4]

    #################################################################x

    print('constrain subsystem 1')
    allc1 = [ (i,0) for i in range(1,Ncc)]
    allc2 = [ (i,1) for i in range(1,Ncc)]

    Naccur = 15
    accur = [ np.exp(u) for u in np.linspace( -6*np.log(10), 1*np.log(10), Naccur) ]
    data = np.zeros(shape= (Naccur, Ncc, *Q.shape))
    for a,acc in enumerate(accur):
        for i in range(Ncc):
            data[a, i] = XX.system(allc1[:i] + allc2[:i], acc)[0]

    filename = 'plot_stoch_c1_v1_Noc.pdf'
    print(f'   variance of subsystem 1 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 0,0]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c1_v1_bin.pdf'
    print(f'   variance of subsystem 1 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 0,0]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c1_v2_Noc.pdf'
    print(f'   variance of subsystem 2 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 2,2]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c1_v2_bin.pdf'
    print(f'   variance of subsystem 2 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 2,2]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c1_v3_Noc.pdf'
    print(f'   variance of subsystem 3 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 4,4]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c1_v3_bin.pdf'
    print(f'   variance of subsystem 3 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 4,4]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    #################################################################x

    print('constrain subsystem 3')
    allc1 = [ (i,4) for i in range(1,Ncc)]
    allc2 = [ (i,5) for i in range(1,Ncc)]

    Naccur = 15
    accur = [ np.exp(u) for u in np.linspace( -6*np.log(10), 1*np.log(10), Naccur) ]
    data = np.zeros(shape= (Naccur, Ncc, *Q.shape))
    for a,acc in enumerate(accur):
        for i in range(Ncc):
            data[a, i] = XX.system(allc1[:i] + allc2[:i], acc)[0]

    filename = 'plot_stoch_c3_v1_Noc.pdf'
    print(f'   variance of subsystem 1 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 0,0]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c3_v1_bin.pdf'
    print(f'   variance of subsystem 1 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 0,0]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c3_v2_Noc.pdf'
    print(f'   variance of subsystem 2 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 2,2]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c3_v2_bin.pdf'
    print(f'   variance of subsystem 2 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 2,2]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c3_v3_Noc.pdf'
    print(f'   variance of subsystem 3 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 4,4]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c3_v3_bin.pdf'
    print(f'   variance of subsystem 3 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 4,4]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining the common cause', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    #################################################################x

    print('constrain subsystem 1 and 2')
    allc1 = [ (i,0) for i in range(1,Ncc)]
    allc2 = [ (i,1) for i in range(1,Ncc)]
    allc3 = [ (i,2) for i in range(1,Ncc)]
    allc4 = [ (i,3) for i in range(1,Ncc)]

    Naccur = 15
    accur = [ np.exp(u) for u in np.linspace( -6*np.log(10), 1*np.log(10), Naccur) ]
    data = np.zeros(shape= (Naccur, Ncc, *Q.shape))
    for a,acc in enumerate(accur):
        data[a, 0] = XX.system([], acc)[0]
        for i in range(Ncc-1):
            data[a, i+1] = XX.system(allc1[:i] + allc2[:i] + allc3[:1] + allc4[:1], acc)[0]

    filename = 'plot_stoch_c12_v1_Noc.pdf'
    print(f'   variance of subsystem 1 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 0,0]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c12_v1_bin.pdf'
    print(f'   variance of subsystem 1 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 0,0]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c12_v2_Noc.pdf'
    print(f'   variance of subsystem 2 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 2,2]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    #ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c12_v2_bin.pdf'
    print(f'   variance of subsystem 2 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 2,2]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c12_v3_Noc.pdf'
    print(f'   variance of subsystem 3 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 4,4]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c12_v3_bin.pdf'
    print(f'   variance of subsystem 3 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 4,4]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    #################################################################x

    print('constrain subsystem 1 and 3')
    allc1 = [ (i,0) for i in range(1,Ncc)]
    allc2 = [ (i,1) for i in range(1,Ncc)]
    allc3 = [ (i,4) for i in range(1,Ncc)]
    allc4 = [ (i,5) for i in range(1,Ncc)]

    Naccur = 15
    accur = [ np.exp(u) for u in np.linspace( -6*np.log(10), 1*np.log(10), Naccur) ]
    data = np.zeros(shape= (Naccur, Ncc, *Q.shape))
    for a,acc in enumerate(accur):
        for i in range(Ncc):
            data[a, i] = XX.system(allc1[:i] + allc2[:i] + allc3[:i] + allc4[:i], acc)[0]

    filename = 'plot_stoch_c13_v1_Noc.pdf'
    print(f'   variance of subsystem 1 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 0,0]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c13_v1_bin.pdf'
    print(f'   variance of subsystem 1 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 0,0]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $x$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c13_v2_Noc.pdf'
    print(f'   variance of subsystem 2 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 2,2]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c13_v2_bin.pdf'
    print(f'   variance of subsystem 2 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 2,2]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $y$ subsystem', fontsize=fontsize)
    ax.set_ylim(-0.1, 1.1*sigma21)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c13_v3_Noc.pdf'
    print(f'   variance of subsystem 3 vs number of constraints --> {filename}')
    fig,ax = plt.subplots(1,1)
    for a in range(Naccur)[:-1:2]:
        ax.plot(range(Ncc), np.abs(data[a, :, 4,4]), label = f'{accur[a]:.2g}', marker = 'x')
    ax.legend(loc='upper right', prop={'size': legendFontSize}, title=r"$\sigma_W^2$ value")
    ax.set_xlabel('number of constraints', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_yscale('log')
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    filename = 'plot_stoch_c13_v3_bin.pdf'
    print(f'   variance of subsystem 3 vs bin size --> {filename}')
    fig,ax = plt.subplots(1,1)
    for i in range(Ncc):
        ax.plot(accur, np.abs(data[:, i, 4,4]), marker='x', label=r'$N_{constr}=$' + f'{i}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='lower right', prop={'size': legendFontSize})
    ax.set_xlabel(r'$\sigma_W^2$', fontsize=fontsize)
    #ax.set_title('Constraining subsystem 1', fontsize=fontsize)
    ax.set_ylabel(f'variance of the $z$ subsystem', fontsize=fontsize)
    fig.savefig(datadir+filename, bbox_inches='tight')
    plt.close(fig)

    #################################################################x
"""
"""
