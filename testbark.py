import numpy as np
# import pandas as pd
class Bark:
    # in order to use the Bark class, we need set the nfft and stimulus frequency related initialization functions first
    # nfft and stimulus frequency are not initialized in the class __init__ function to save the class inital time
    # Bark.set_nfft(nfft)
    # Bark.set_stimulus_freq(stimulus_freq)

    def __init__(self, nfft=2048, fs=48000, nfilts=25, refmic_sens=-40,stimulus_freq=1000):
        self.bgnoise=28 # background noise level in dBA
        self.NF = nfft
        self.Nadv = self.NF // 2
        self.Fs = fs
        #Number of critical bands:
        self.Nc = nfilts
        # self.P = np.zeros((2, self.Nc))
        # self.Rn = np.zeros((self.Nc))
        # self.Rd = np.zeros((self.Nc))
        # self.PC = np.zeros((2, self.Nc))

        #self.max_freq = maxfreq
        self.max_freq = min(int(fs/2),20000)
        self.Emin = 1e-12
        self.refmic=refmic_sens  #rms value of dBFs, if refmic_sens=-3 (Vrms) means the fullscal signal 0dBFs(Vp) 
        self.stifreq=stimulus_freq

        #Critical band parameters for the FFT model, for Basic Version:
        self.fc, self.fl, self.fu = self.CB_filters()

        #Critical band parameters for the FFT model, for Basic Version:
        self.dz = 0.25  # affect the mask curve slope on the right side, + self.dz*Lp_stimulus 
        self.e = 0.4  # e is a compression mixing exponent (0.1< e < 2). 
        # In PEAQ its value is set to 0.4, masker in the excitation bark will be about 10dB below the stimulus
        # if set to 2, masker in the excitation bark will be equal to the stimulus level 
        # if set to 0.1, masker in the excitation bark will be more than 50dB below the stimulus

        # Allocate storage
        self.Eb = np.zeros((2, self.Nc))
        self.Xw2 = np.zeros((2, self.NF//2+1))
        # self.XwN2 = np.zeros(self.NF//2+1)
        self.E = np.zeros(self.Eb.shape)
        self.Es = np.zeros((2, self.Nc))

        #find stimulus frequency's critical band number 
        # self.sti_cbindex=self.Sti_CB(self.stifreq)
        # self.sti_cbfreq=self.fc[self.sti_cbindex]

        #find the H harmonics frequency's critical band number 
        # self.H_cbindex=self.Sti_CB(self.stifreq)

        # Compute the forward Bark transform weight matrix
        # self.W_Bark = self.fft2barkmx_peaq()
        
        #Precompute frequency vector:
        self.f = np.linspace (0, int(self.Fs/2), int(self.NF/2)+1)
        #Outer and middle ear weighting:
        self.W_Ear = self.PQWOME (self.f)
        
        # absolute listening threshold 
        self.Abth=self.Absolute_TH()

		#Internal Noise:
        # Need scal the dBSPL ?
        self.EIN = self.PQIntNoise(self.fc)
        #Precompute normalization for frequency spreading:
        self.Bs = self.PQ_SpreadCB(np.ones(self.Nc), np.ones(self.Nc))
        # check FLAG, False means first operation
        self.check_PQmodPatt = False


        # rawdata=pd.DataFrame(data=self.Bs)
        # rawdata.to_csv('./Bs.csv')

        # print ("The forward and backward Bark transform matrices have been precomputed - ready to go!")
        # print ("The transform is for ", self.Nc, " filters, a sampling rate of ", self.Fs, \
        #       " and FFT size of ", self.NF)
    
    def set_stimulus_freq(self,stimulus_freq):
        self.stifreq=stimulus_freq
        self.sti_cbindex=self.Sti_CB(self.stifreq)
        self.sti_cbfreq=self.fc[self.sti_cbindex]
        # self.H_cbindex=self.Sti_CB(1*self.stifreq)
        self.H_cbindex=self.sti_cbindex

    def set_nfft(self,nfft):
        self.NF = nfft
        self.Nadv = self.NF // 2
        self.f = np.linspace (0, int(self.Fs/2), int(self.NF/2)+1)
        self.W_Ear = self.PQWOME (self.f)
        self.Xw2 = np.zeros((2, self.NF//2+1))
        # self.XwN2 = np.zeros(self.NF//2+1)
        self.W_Bark = self.fft2barkmx_peaq() 

    def Sti_CB(self,stimulus_freq):
       # find the stimlus critical band index 
        for i in range(len(self.fu)):
            if self.fu[i]-stimulus_freq>0:
                break 
        return i 
    
 	#Internal noise:
    def PQIntNoise (self, f):
        INdB = 1.456 * (f / 1000.)**(-0.8)   # dBSPL scale ? 
        # Convert to normalized dBFs
        # full scale signal(normalize to -1~1, Vp=1,Vrms=0.707,) 
        # correspond to -3dBFs(rms) and (94-refmic)dBSPL
        INdBFs=-94+self.refmic+INdB 
        EIN = 10**(INdBFs / 10.)
        return EIN

    def fft2barkmx_peaq(self):
        #FFT to bark matrix using PEAQ method 
        nfft = self.NF
        nfilts  = self.Nc
        fs = self.Fs    
        df = float(fs)/nfft
        W = np.zeros((nfilts, int(nfft/2)+1))

        # The following k,i loop script cost about 0.3s for one cycle,need to be optimized
        # for k in range(int(nfft/2)+1):
        #     for i in range(nfilts):
        #         temp = (np.amin([self.fu[i], (k+0.5)*df]) - np.amax([self.fl[i], (k-0.5)*df])) / df
        #         W[i,k] = np.amax([0, temp])
        
        # Rewite the k,i loop to i loop, cost about 0.01s for one cycle
        # k=np.arange(int(nfft/2)+1)
        # for i in range(nfilts):
        #     R_bound=(k+0.5)*df
        #     R_bound[R_bound>self.fu[i]]=self.fu[i]
        #     L_bound=(k-0.5)*df
        #     L_bound[L_bound<self.fl[i]]=self.fl[i]   
        #     temp = (R_bound-L_bound) / df
        #     temp[temp<0]=0
        #     W[i] = temp      

        #rewrite the i loop to vector operation, cost about 0.018s for 100 cycles 
        k=np.arange(int(nfft/2)+1)
        R_bound=np.minimum(self.fu[:,np.newaxis],(k+0.5)*df)
        L_bound=np.maximum(self.fl[:,np.newaxis],(k-0.5)*df)
        temp = (R_bound-L_bound) / df
        temp[temp<0]=0
        W = temp
        # rawdata=pd.DataFrame(data=W)
        # rawdata.to_csv('./W.csv')
        return W

    def PQWOME(self,f):
    # Generate the weighting for the outer & middle ear filtering
    # output a magnitude-squared vector
        
        # The following loop script cost about 0.29s for 100 cycles,need to be optimized
        # N = len(f)
        # W2 = np.zeros(N)
        # for k in range(N-1):
        #     fkHz = float(f[k+1])/1000
        #     AdB = -2.184 * fkHz**(-0.8) + 6.5 * np.exp(-0.6 * (fkHz - 3.3)**2) - 0.001 * fkHz**(3.6)
        #     W2[k+1] = 10**(AdB / 10)

        # Rewite the loop to vector operation, cost about 0.029s for 100 cycles
        # fkHz = f[1:]/1000
        fkHz=f/1000
        fkHz[0]=self.Emin #avoid the log(0) error
        AdB = -2.184 * fkHz**(-0.8) + 6.5 * np.exp(-0.6 * (fkHz - 3.3)**2) - 0.001 * fkHz**(3.6)
        W2=10**(AdB / 10)
        return W2    

    def CB_filters(self):
        #Critical band filters for creation of the PEAQ FFT model
        #(Basic Version) forward Bark domain transform weight matrix
        # if self.Nc > 100:
        #     fl = np.array([  80.000,   103.445,   127.023,   150.762,   174.694, \
        #         198.849,   223.257,   247.950,   272.959,   298.317, \
        #         324.055,   350.207,   376.805,   403.884,   431.478, \
        #         459.622,   488.353,   517.707,   547.721,   578.434, \
        #         609.885,   642.114,   675.161,   709.071,   743.884, \
        #         779.647,   816.404,   854.203,   893.091,   933.119, \
        #         974.336,  1016.797,  1060.555,  1105.666,  1152.187, \
        #         1200.178,  1249.700,  1300.816,  1353.592,  1408.094, \
        #         1464.392,  1522.559,  1582.668,  1644.795,  1709.021, \
        #         1775.427,  1844.098,  1915.121,  1988.587,  2064.590, \
        #         2143.227,  2224.597,  2308.806,  2395.959,  2486.169, \
        #         2579.551,  2676.223,  2776.309,  2879.937,  2987.238, \
        #         3098.350,  3213.415,  3332.579,  3455.993,  3583.817, \
        #         3716.212,  3853.817,  3995.399,  4142.547,  4294.979, \
        #         4452.890,  4616.482,  4785.962,  4961.548,  5143.463, \
        #         5331.939,  5527.217,  5729.545,  5939.183,  6156.396, \
        #         6381.463,  6614.671,  6856.316,  7106.708,  7366.166, \
        #         7635.020,  7913.614,  8202.302,  8501.454,  8811.450, \
        #         9132.688,  9465.574,  9810.536, 10168.013, 10538.460, \
        #         10922.351, 11320.175, 11732.438, 12159.670, 12602.412, \
        #         13061.229, 13536.710, 14029.458, 14540.103, 15069.295, \
        #         15617.710, 16186.049, 16775.035, 17385.420 ])
        #     fc = np.array([  91.708,   115.216,   138.870,   162.702,   186.742, \
        #         211.019,   235.566,   260.413,   285.593,   311.136, \
        #         337.077,   363.448,   390.282,   417.614,   445.479, \
        #         473.912,   502.950,   532.629,   562.988,   594.065, \
        #         625.899,   658.533,   692.006,   726.362,   761.644, \
        #         797.898,   835.170,   873.508,   912.959,   953.576, \
        #         995.408,  1038.511,  1082.938,  1128.746,  1175.995, \
        #         1224.744,  1275.055,  1326.992,  1380.623,  1436.014, \
        #         1493.237,  1552.366,  1613.474,  1676.641,  1741.946, \
        #         1809.474,  1879.310,  1951.543,  2026.266,  2103.573, \
        #         2183.564,  2266.340,  2352.008,  2440.675,  2532.456, \
        #         2627.468,  2725.832,  2827.672,  2933.120,  3042.309, \
        #         3155.379,  3272.475,  3393.745,  3519.344,  3649.432, \
        #         3784.176,  3923.748,  4068.324,  4218.090,  4373.237, \
        #         4533.963,  4700.473,  4872.978,  5051.700,  5236.866, \
        #         5428.712,  5627.484,  5833.434,  6046.825,  6267.931, \
        #         6497.031,  6734.420,  6980.399,  7235.284,  7499.397, \
        #         7773.077,  8056.673,  8350.547,  8655.072,  8970.639, \
        #         9297.648,  9636.520,  9987.683, 10351.586, 10728.695, \
        #         11119.490, 11524.470, 11944.149, 12379.066, 12829.775, \
        #         13294.850, 13780.887, 14282.503, 14802.338, 15341.057, \
        #         15899.345, 16477.914, 17077.504, 17690.045 ])
        #     fu = np.array([ 103.445,   127.023,   150.762,   174.694,   198.849, \
        #         223.257,   247.950,   272.959,   298.317,   324.055, \
        #         350.207,   376.805,   403.884,   431.478,   459.622, \
        #         488.353,   517.707,   547.721,   578.434,   609.885, \
        #         642.114,   675.161,   709.071,   743.884,   779.647, \
        #         816.404,   854.203,   893.091,   933.113,   974.336, \
        #         1016.797,  1060.555,  1105.666,  1152.187,  1200.178, \
        #         1249.700,  1300.816,  1353.592,  1408.094,  1464.392, \
        #         1522.559,  1582.668,  1644.795,  1709.021,  1775.427, \
        #         1844.098,  1915.121,  1988.587,  2064.590,  2143.227, \
        #         2224.597,  2308.806,  2395.959,  2486.169,  2579.551, \
        #         2676.223,  2776.309,  2879.937,  2987.238,  3098.350, \
        #         3213.415,  3332.579,  3455.993,  3583.817,  3716.212, \
        #         3853.348,  3995.399,  4142.547,  4294.979,  4452.890, \
        #         4643.482,  4785.962,  4961.548,  5143.463,  5331.939, \
        #         5527.217,  5729.545,  5939.183,  6156.396,  6381.463, \
        #         6614.671,  6856.316,  7106.708,  7366.166,  7635.020, \
        #         7913.614,  8202.302,  8501.454,  8811.450,  9132.688, \
        #         9465.574,  9810.536, 10168.013, 10538.460, 10922.351, \
        #         11320.175, 11732.438, 12159.670, 12602.412, 13061.229, \
        #         13536.710, 14029.458, 14540.103, 15069.295, 15617.710, \
        #         16186.049, 16775.035, 17385.420, self.max_freq ])
        # else:
        #     fl = np.array([  0., 100., 200., 300., 400., \
        #                 510., 630., 770., 920., 1080., \
        #                 1270., 1480., 1720., 2000., 2320., \
        #                 2700., 3150., 3700., 4400., 5300.,\
        #                 6400., 7700., 9500., 12000., 15500.])
        #     fc = np.array([  50., 150., 250., 350., 450., \
        #                     570., 700., 840., 1000., 1170., \
        #                     1370., 1600., 1850., 2150., 2500., \
        #                     2900., 3400., 4000., 4800., 5800.,\
        #                     7000., 8500., 10500., 13600., 19500.])
        #     fu = np.array([  100., 200., 300., 400., 510., \
        #                     630., 770., 920., 1080., 1270., \
        #                     1480., 1720., 2000., 2320., 2700., \
        #                     3150., 3700., 4400., 5300., 6400., \
        #                     7700., 9500., 12000., 15500.,self.max_freq])

        # critical band = 25 
        fl = np.array([  0., 100., 200., 300., 400., \
                    510., 630., 770., 920., 1080., \
                    1270., 1480., 1720., 2000., 2320., \
                    2700., 3150., 3700., 4400., 5300.,\
                    6400., 7700., 9500., 12000., 15500.])
        fc = np.array([  50., 150., 250., 350., 450., \
                        570., 700., 840., 1000., 1170., \
                        1370., 1600., 1850., 2150., 2500., \
                        2900., 3400., 4000., 4800., 5800.,\
                        7000., 8500., 10500., 13600., 19500.])
        # fu = np.array([  100., 200., 300., 400., 510., \
        #                 630., 770., 920., 1080., 1270., \
        #                 1480., 1720., 2000., 2320., 2700., \
        #                 3150., 3700., 4400., 5300., 6400., \
        #                 7700., 9500., 12000., 15500.,self.max_freq])
        fu = np.array([  100., 200., 300., 400., 510., \
                        630., 770., 920., 1080., 1270., \
                        1480., 1720., 2000., 2320., 2700., \
                        3150., 3700., 4400., 5300., 6400., \
                        7700., 9500., 12000., 15500.,22000])
        return fc, fl, fu

    def Absolute_TH(self):
        if self.Nc > 100: 
            #use 109 critical band 
            pass
        else:
            #use 25 critical band 
            # dBSPL value of the threshold of hearing , need ref mic sensitivity info to convert the test data from dBFs to dBSPL
            a_th=np.array([ 55.0, 30.0, 20.5, 16.0, 11.0, \
                            9.0, 8.0, 8.5, 9.0, 10.0, \
                            11.2, 11.8, 12.4, 13.0,  14.0, \
                            13.5, 13.1, 12.8, 12.0, 13.9, \
                            15.6,  16.9, 20.5, 25.0, 55.0])
        return(a_th)

    def PQspreadCB(self, E):
		# Spread an excitation vector (pitch pattern) - FFT model
		# Both E and Es are powers	  
        Es = self.PQ_SpreadCB(E, self.Bs)
        return Es

    def PQ_SpreadCB(self, E, Bs):
        e = self.e 
        # e is a compression mixing exponent (0< e < 2). 
        # In PEAQ its value is set to 0.4   
                   
		# Initialize arrays for storage. These values are used
		# in each iteration (summed over, multiplied, raised to
		# powers, etc.) when computing the spread Bark-domain
		# energy Es.
		#
		# aUCEe is for the product of bin-dependent (index l)
		# term aC, energy-dependent (E) term aE, and
		# term aU.
		#
		# Ene is (E[l]/A(l,E[l]))^e, stored for each index l
		#
		# Es is the overall spread Bark-domain energy
		#
        aUCEe = np.zeros(self.Nc)
        Ene = np.zeros(self.Nc)
        Es = np.zeros(self.Nc)
		
		# Calculate energy-dependent terms
        aL = 10**(2.7*self.dz)
        # for l in range(self.Nc):
        #     aUC = 10**((-2.4 - 23/self.fc[l])*self.dz)
        #     aUCE = aUC * (E[l]**(0.2*self.dz))
        #     gIL = (1 - aL**(-1*(l+1))) / (1 - aL**(-1))
        #     gIU = (1 - (aUCE)**(self.Nc-l)) / (1 - aUCE)
        #     En = E[l] / (gIL + gIU - 1)
        #     aUCEe[l] = aUCE**(e)
        #     Ene[l] = En**(e)

        # rewrite the above for loop to vectorize the code and save time
        aUC = 10**((-2.4 - 23/self.fc)*self.dz)
        aUCE = aUC * (E**(0.2*self.dz))
        gIL = (1 - aL**(-1*(np.arange(self.Nc)+1))) / (1 - aL**(-1))
        gIU = (1 - (aUCE)**(self.Nc-np.arange(self.Nc))) / (1 - aUCE)
        En = E / (gIL + gIU - 1)
        aUCEe = aUCE**(e)
        Ene = En**(e)

        # Lower spreading
        Es[self.Nc-1] = Ene[self.Nc-1]
        aLe = aL**(-1*e)

        for i in range((self.Nc-2),-1,-1):
            Es[i] = aLe*Es[i+1] + Ene[i]
        # rewrite the above for loop to vectorize the code and save time
        # the following code is wrong, need shift and acummulate to get the right result
        # will study the code later
        # Es[0:(self.Nc-1)] = aLe*Es[1:self.Nc] + Ene[0:(self.Nc-1)]


        # Upper spreading (i > m)
        for i in range(0,(self.Nc-1)):
            r = Ene[i]
            a = aUCEe[i]
            for l in range((i+1),self.Nc):
                r = r*a
                Es[l] = Es[l] + r
        #rewrite the above i loop to vectorize the code and save time
     

        # Normalize the values by the normalization factor
        # for i in range(0,self.Nc):
        #     Es[i] = (Es[i]**(1/e)) / Bs[i]
        #rewrite the above for loop to vectorize the code and save time
        Es = (Es**(1/e)) / Bs

        return Es

    # def maskthreshold(self,barkspecdB):
    #     lv=barkspecdB[self.sti_cbindex]-10
    #     fv=self.stifreq
    #     svh=-24-230/fv+0.2*lv

    #     barkthreshold=np.ones(self.Nc)
    #     # check if len(barkspecdB)=self.NC
    #     # .......
    #     barkthreshold[self.sti_cbindex]=lv
    #     for i in range(self.sti_cbindex,self.Nc-1):
    #         barkthreshold[i+1]=barkthreshold[i]+svh

    #     for i in range(0,self.sti_cbindex):
    #         barkthreshold[i]=max(0,barkthreshold[self.sti_cbindex]-(self.sti_cbindex-i)*27)
    #     return barkthreshold

    def fft2bark(self, X2,Harmonic_order=2):
        # X2 = np.zeros((2,self.NF//2+1))
        # X2[0,:] = ref_sig_x2  # reference signal power spectrum
        # X2[1,:] = test_sig_x2  # test signal power spectrum 

    
        # Outer and middle ear filtering
        self.Xw2[0,:] = self.W_Ear * X2[0,:]   # ref signal 
        # self.Xw2[1,:] = self.W_Ear * X2[1,:]   # test signal 
        self.Xw2[1,:] = self.W_Ear * X2[1,:]   # test signal 
        # Form the difference magnitude signal
        # self.XwN2 = self.Xw2[0,:] - 2*np.sqrt(self.Xw2[0,:]*self.Xw2[1,:]) + self.Xw2[1,:]

        # Group into partial critical bands
        self.Eb[0,:] = np.dot(self.W_Bark,self.Xw2[0,:]) 
        self.Eb[1,:] = np.dot(self.W_Bark,self.Xw2[1,:]) 
        # self.EbN = np.dot(self.W_Bark,self.XwN2) 

        self.Eb[self.Eb<self.Emin] = self.Emin
        # self.EbN[self.EbN<self.Emin] = self.Emin

        # add the internal nosie
        self.E[0,:] = self.Eb[0,:] + self.EIN
        self.E[1,:] = self.Eb[1,:] + self.EIN

        #only the funderment/stimulus is included in maskee calculation 
        bark_4ref=self.E[0,:]+self.Emin
        bark_4mask=self.E[1,:]+self.Emin

        msk_start= max(0,self.sti_cbindex-1)
        bark_4ref[0:msk_start]=1e-12
        bark_4mask[0:msk_start]=1e-12
        msk_end_r=min(self.H_cbindex+1,self.Nc-1)
        bark_4ref[msk_end_r:-1]=1e-12
        bark_4mask[msk_end_r:-1]=1e-12

        # # Critical band spreading => "Unsmeared (in time) excitation patterns"
        # self.Es[0,:] = self.PQspreadCB(self.E[0,:])
        # self.Es[1,:] = self.PQspreadCB(self.E[1,:])

        # only the funderment/stimulus to 2nd harmonics fre bins are included in maskee calculation
        Ref = self.PQspreadCB(bark_4ref) 
        Masker = self.PQspreadCB(bark_4mask) 

        Ref2CB_dB=10*np.log10(self.E[0,:])-self.refmic+94
        Ref_Masker_dB=10*np.log10(Ref)-self.refmic+94
        # Noise_dB=Ref2CB_dB-Ref_Masker_dB
        # Noise_dB[Noise_dB<0]=0
        # Noise_dB[Ref2CB_dB<self.Abth]=0
        # Noise_dB[msk_start-1:msk_end_r]=0        # For noise loundess calculation
        # Noise_dB[0:2]=0  # set the 200Hz and below energy to zero
        
        Noise_dB_diff=Ref2CB_dB-Ref_Masker_dB
        Noise_dB=Ref2CB_dB.copy()
        # Noise_dB_avg=np.mean(np.sort(Noise_dB[2:-1])[1:-1])
        bgnoise=self.bgnoise+np.random.rand()*0.1
        Noise_dB[Ref2CB_dB<self.Abth]=bgnoise
        Noise_dB[Noise_dB_diff<0]=bgnoise
        Noise_dB[max(msk_start-1,0):min(msk_end_r+2,self.Nc)]=bgnoise
        Noise_dB[0:2]=0  # set the 200Hz and below energy to zero
        # print(msk_start)
        # print(msk_end_r)
        # print(Noise_dB)
        Test2CB_dB=10*np.log10(self.E[1,:])-self.refmic+94
        Masker_dB=10*np.log10(Masker)-self.refmic+94
        NL_dB=Test2CB_dB-Masker_dB
        NL_dB[NL_dB<0]=0
        NL_dB[Test2CB_dB<self.Abth]=0
        NL_dB[0:msk_end_r]=0

        #find the user specified start harmonics frequency's critical band number 
        H_cbindex=self.Sti_CB(self.stifreq*Harmonic_order)
        # Noise_dB[0:H_cbindex]=0
        NL_dB[0:H_cbindex]=0
        
        threshold=1
        NT=NL_dB[NL_dB>threshold]
        NR=Noise_dB[Noise_dB>threshold]

        # remove the  min value bin and the possible tone like bin 
        NR_mean=np.mean(np.sort(NR)) 
        NR_above_mean=NR[NR>NR_mean]
        if len(NR_above_mean)<2:
            NR_r = np.sort(NR)[1:-1]
        else:
            NR_r = np.sort(NR)[1:]
            

        Loudness=max(10*np.log10(max(sum(10**(NT/10)),self.Emin)),1) # Buzz loudness
        Noise_Loudness=max(10*np.log10(max(sum(10**(NR_r/10)),self.Emin))/1.5,bgnoise) # noise loudness 
        # Noise_Loudness=max(np.mean(NR_r),bgnoise) # noise loudness in dB
        # Noise_Loudness=np.mean(NR_r) # noise loudness in dB
      

        # EP = self.PQadapt(self.Es[0,:], self.Es[1,:], 'FFT')
        # M, ERavg = self.PQmodPatt()
        # NLoud_NL = self.PQmovNLoudB(M, EP) # Partial Noise Loudness 
        # NLoud_NL_dB=94-self.refmic+10*np.log10(NLoud_NL/10)
        # return self.E,self.Es,self.NLoud_NL

        return Ref2CB_dB,Ref_Masker_dB,Loudness,Noise_Loudness

    def fft2barkall(self, X2):
        # X2 = np.zeros((2,self.NF//2+1))
        # X2[0,:] = ref_sig_x2  # reference signal power spectrum
        # X2[1,:] = test_sig_x2  # test signal power spectrum 

    
        # Outer and middle ear filtering
        # self.Xw2[0,:] = self.W_Ear * X2[0,:]   # ref signal 
        self.Xw2[1,:] = self.W_Ear * X2[1,:]   # test signal 

        # Form the difference magnitude signal
        # self.XwN2 = self.Xw2[0,:] - 2*np.sqrt(self.Xw2[0,:]*self.Xw2[1,:]) + self.Xw2[1,:]

        # Group into partial critical bands
        # self.Eb[0,:] = np.dot(self.W_Bark,self.Xw2[0,:]) 
        self.Eb[1,:] = np.dot(self.W_Bark,self.Xw2[1,:]) 
        # self.EbN = np.dot(self.W_Bark,self.XwN2) 

        self.Eb[self.Eb<self.Emin] = self.Emin
        # self.EbN[self.EbN<self.Emin] = self.Emin

        # add the internal nosie
        # self.E[0,:] = self.Eb[0,:] + self.EIN
        self.E[1,:] = self.Eb[1,:] + self.EIN

        bark_4mask=self.E[1,:]+self.Emin
  
        Masker = self.PQspreadCB(bark_4mask) 
        
        Test2CB_dB=10*np.log10(self.E[1,:])-self.refmic+94
        Masker_dB=10*np.log10(Masker)-self.refmic+94
        NL_dB=Test2CB_dB-Masker_dB
        NL_dB[NL_dB<0]=0
        NL_dB[Test2CB_dB<self.Abth]=0
        
        #find the user specified start harmonics frequency's critical band number 
        H_cbindex=self.Sti_CB(self.stifreq)
        NL_dB[0:H_cbindex]=0
        threshold=1
        NT=NL_dB[NL_dB>threshold]

        Loudness=max(10*np.log10(max(sum(10**(NT/10)),self.Emin)),1) # loudness in dB


        # EP = self.PQadapt(self.Es[0,:], self.Es[1,:], 'FFT')
        # M, ERavg = self.PQmodPatt()
        # NLoud_NL = self.PQmovNLoudB(M, EP) # Partial Noise Loudness 
        # NLoud_NL_dB=94-self.refmic+10*np.log10(NLoud_NL/10)
        # return self.E,self.Es,self.NLoud_NL
        return Test2CB_dB,Masker_dB,Loudness
    
    # def fft2bark_EarWeightOff(self, spectrum):
    #     bark_spectrum = np.dot(self.W_Bark,spectrum) 
    #     bark_spectrum[bark_spectrum<self.Emin] = self.Emin
    #     # add the internal nosie
    #     bark_spectrum=bark_spectrum+self.EIN
    #     return bark_spectrum

    # def fft2bark_InterNoiseOff(self, spectrum):
    #     spectrum_inEar= self.W_Ear*spectrum
    #     bark_spectrum = np.dot(self.W_Bark,spectrum_inEar) 
    #     bark_spectrum[bark_spectrum<self.Emin] = self.Emin
    #     # skip the internal nosie adding 
    #     # bark_spectrum=bark_spectrum+self.EIN
    #     return bark_spectrum


    # # For distortion noise loudness calculation
    # def PQadapt(self, EhsR, EhsT, Mod='FFT'):
    #         if Mod != 'FFT':
    #             raise ValueError(f'Mod only supports FFT, but {Mod}')
            
    #         Fs = self.Fs
    #         Fss = Fs / self.Nadv
    #         t100 = 0.050
    #         tmin = 0.008
    #         a, b = self.PQtConst(t100, tmin, self.fc, Fss)
    #         M1, M2 = 3, 4

    #         EP = np.zeros((2, self.Nc))
    #         R = np.zeros((2, self.Nc))

    #         self.P = np.expand_dims(a,-2) * self.P + np.expand_dims(b,-2) * np.stack([EhsR, EhsT])
    #         sn = np.sum(np.sqrt(self.P[...,0,:] * self.P[...,1,:]), -1)
    #         sd = np.sum(self.P[...,1,:], -1)

    #         CL = (sn / sd) ** 2
    #         cond = CL > 1
    #         EP[0] = np.where(cond, EhsR / CL, EhsR)
    #         EP[1] = np.where(cond, EhsT, EhsT * CL)

    #         self.Rn = a * self.Rn + EP[1] * EP[0]
    #         self.Rd = a * self.Rd + EP[0] ** 2

    #         cond = self.Rn >= self.Rd
    #         R[0] = np.where(cond, 1, self.Rn / self.Rd)
    #         R[1] = np.where(cond, self.Rd / self.Rn, 1)
            
    #         for m in range(self.Nc):
    #             iL = max(m - M1, 0)
    #             iU = min(m + M2, self.Nc-1)
    #             s1 = np.sum(R[0,iL:iU+1], -1)
    #             s2 = np.sum(R[1,iL:iU+1], -1)

    #             self.PC[0,m] = a[m] * self.PC[0,m] + b[m] * s1 / (iU-iL+1)
    #             self.PC[1,m] = a[m] * self.PC[1,m] + b[m] * s2 / (iU-iL+1)

    #             EP[0,m] *= self.PC[0,m]
    #             EP[1,m] *= self.PC[1,m]
    #         return EP

    # def PQmodPatt(self):
    #     Nadv = self.NF//2
    #     Fss = float(self.Fs)/Nadv
    #     tau_100 = 0.050
    #     tau_min = 0.008
    #     alpha, beta = self.PQtConst(tau_100, tau_min, self.fc, Fss)
    #     if self.check_PQmodPatt == False:
    #         self.DE = np.zeros((2, self.Nc))
    #         self.Ese = np.zeros((2, self.Nc))
    #         self.Eavg = np.zeros((2, self.Nc))
    #         self.check_PQmodPatt = True
        
    #     e = 0.3
    #     Ee = self.Es ** e
    #     alpha, beta = alpha[None], beta[None]
    #     self.DE = alpha * self.DE + beta * Fss * np.abs(Ee - self.Ese)
    #     self.Eavg = alpha * self.Eavg + beta * Ee
    #     self.Ese = Ee
    #     M = self.DE / (1 + self.Eavg / e)
    #     ERavg = self.Eavg[0]
    #     return M, ERavg
    
    # def PQmovNLoudB(self, M, EP):
    #         alpha = 1.5
    #         TF0 = 0.15
    #         S0 = 0.5
    #         NLmin = 0
    #         e = 0.23
    #         s = 0

    #         sref = TF0 * M[0] + S0
    #         test = TF0 * M[1] + S0
    #         beta = np.exp(-alpha * (EP[1] - EP[0]) / EP[0])
    #         tmp = test * EP[1] - sref * EP[0]
    #         a = np.maximum(tmp, np.zeros_like(tmp))
    #         b = self.EIN + sref * EP[0] * beta
    #         s = np.sum((self.EIN / test) ** e * ((1 + a / b) ** e - 1))
    #         NL = (24 / self.Nc) * s
    #         if NL < NLmin:
    #             return 0
    #         return NL

    # def PQtConst(self, tau_100, tau_min, fc, Fss):
    #     # Tau values in units of seconds
    #     #tau_100 = 0.030
    #     #tau_min = 0.008
        
    #     tau = np.zeros(len(fc))
    #     alpha = np.zeros(len(fc))
        
    #     tau = tau_min + (np.divide(float(100),fc))*(tau_100 - tau_min)
    #     alpha = np.exp(np.divide(-1./Fss,tau))
    #     beta = 1. - alpha

    #     return alpha, beta

    # For EHS(error harmonics structure) calculation 
    def PQHannWin(self, NF):
        n = np.arange(0, NF)
        hw = 0.5*(1-np.cos(2*np.pi*n/(NF-1)))
        return hw

    def PQRFFT (self, x, N, ifn):
            # Calculate the DFT of a real N-point sequence or the inverse
            # DFT corresponding to a real N-point sequence.
            # ifn > 0, forward transform
            #          input x(n)  - N real values
            #          output X(k) - The first N/2+1 points are the real
            #            parts of the transform, the next N/2-1 points
            #            are the imaginary parts of the transform. However
            #            the imaginary part for the first point and the
            #            middle point which are known to be zero are not
            #            stored.
            # ifn < 0, inverse transform
            #          input X(k) - The first N/2+1 points are the real
            #            parts of the transform, the next N/2-1 points
            #            are the imaginary parts of the transform. However
            #            the imaginary part for the first point and the
            #            middle point which are known to be zero are not
            #            stored. 
            #          output x(n) - N real values

            if (ifn > 0):
                X = np.fft.fft (x, N)
                XR = np.real(X[0:N//2+1])
                XI = np.imag(X[1:N//2-1+1])
                X = np.concatenate([XR, XI])
                return X
            else:
                raise Exception('ifft Not Implemented Yet -SW')

    def PQRFFTMSq(self, X, N):
        # Calculate the magnitude squared frequency response from the
        # DFT values corresponding to a real signal (assumes N is even)

        X2 = np.zeros(N//2+1)

        X2[0] = X[0]**2

        # for k in range(N//2-1):
        #     X2[k+1] = X[k+1]**2 + X[N//2+k+1]**2

        # rewrite the above k loop as a vector operation
        X2[1:N//2-1+1] = X[1:N//2-1+1]**2 + X[N//2+1:N//2-1+N//2+1]**2

        X2[N//2] = X[N//2]**2
        return X2

    def PQ_Corr(self, D, NL, M): # DFT-based operation in original matlab code
        M = M.astype(int)
        NL = NL.astype(int)

        C = np.zeros(NL)
        # for i in range(NL):
        #     s = 0
        #     for j in range(M):
        #         s += D[...,j] * D[...,i+j]
        #     C[i] = s
        
        # rewrite the above j loop as a vector operation
        # for i in range(NL):
        #    C[i] = np.sum(D[..., :M] * D[..., i:i+M], axis=-1).sum()
        
        # rewrite the above loop as a einsum operation to make it more faster
        for i in range(NL):
            C[i] = np.einsum('...i,...i->...', D[..., :M], D[..., i:i+M]).sum()
        
        return C
    
    def PQ_NCorr(self, C, D, NL, M):
        NL = NL.astype(int)
        M = M.astype(int)
        Cn = np.zeros((NL,))

        s0 = C[0]
        sj = s0
        Cn[0] = 1
        for i in range(1, NL):
            sj += (D[i+M-1] ** 2 - D[i-1] ** 2)
            d = s0 * sj
            if d <= 0:
                Cn[i] = 1
            else:
                Cn[i] = C[i] / d ** 0.5
        return Cn
        
    def PQmovEHS(self, max_harmonic_order,Xd):
            
            # X2 = np.zeros((2,self.NF//2+1))
            # X2[0,:] = ref_sig_x2  # reference signal power spectrum
            # X2[1,:] = test_sig_x2  # test signal power spectrum 
            # NF = 2048
            if max_harmonic_order < 5:
                EHS=1 
            else:    
                # Xd the FFT spectrum of the harmonic structure of the signal in dB (ref_sig_x2-test_sig_x2)
                NF = self.NF  #number of FFT size 
                # Fs = 48000
                Fs = self.Fs
                Fc = self.stifreq
                Fmax = 9000   
                NL = 2**(self.PQ_log2(NF * Fmax / Fs))
                M = NL
                # print(M)
                Hw = (1 / M) * (8 / 3) ** 0.5 * self.PQHannWin(M)

                # EnThr = 8000
                # kmax = NL + M - 1

                # xR, xT = np.copy(xR).astype(np.float64), np.copy(xT).astype(np.float64)

                # EnRef  = np.matmul(xR[Nadv:NF+1], xR[Nadv:NF+1].T)
                # EnTest = np.matmul(xT[Nadv:NF+1], xT[Nadv:NF+1].T)

                # if EnRef < EnThr and EnTest < EnThr:
                #     return -1

                # D1 = np.log(X2[1] / X2[0])

                # D1 = Xd
                # D1[D1<0]=0
                # D1 = np.log(X2[1])

                # D = D1.copy()
                # Harmonic_order=max(Harmonic_order,1) # set the lowest harmonic order =1
                # D_len=len(D1)
                # l=int(2*(Harmonic_order-0.5)*D_len*Fc/Fs)   # specified harmonic freq FFT start index 

                # D[0:D_len-l-1]=D1[l:-1]
                # D[D_len-l:]=0.01 #the value may replaced by small value(dB). 

            

                C = self.PQ_Corr(Xd, NL, M)

                Cn = self.PQ_NCorr(C, Xd, NL, M)
                Cnm = (1 / NL) * np.sum(Cn[:NL.astype(int)+1])

                Cw = Hw * (Cn - Cnm)

                cp = self.PQRFFT(Cw, NL.astype(int), 1)
                c2 = self.PQRFFTMSq(cp, NL.astype(int))

                EHS = self.PQ_FindPeak(c2, (NL/2+1).astype(int))
                EHS=1000*max(EHS,0.0001)

            return EHS

    @staticmethod
    def PQ_log2(x):
        res = np.zeros_like(x)
        m = 1
        while m < x:
            res = res + 1
            m *= 2
        return res - 1
    
    # @staticmethod
    # def PQ_FindPeak(c2, N):
    #     cprev = c2[0]
    #     cmax = 0
    #     for n in range(1, N):
    #         if c2[n] > cprev and c2[n] > cmax:
    #             cmax = c2[n]
    #     return cmax
    
