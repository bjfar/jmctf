import sys
sys.path.append("..") # Adds higher directory to python modules path.
import yaml
from analysis import ColliderAnalysis, collider_analyses_from_long_YAML

# # All
# "ATLAS_13TeV_4LEP_36invfb",
# "ATLAS_13TeV_RJ3L_2Lep2Jets_36invfb",
# "ATLAS_13TeV_RJ3L_3Lep_36invfb",
# "ATLAS_13TeV_MultiLEP_2Lep0Jets_36invfb",
# "ATLAS_13TeV_MultiLEP_2LepPlusJets_36invfb",
# "ATLAS_13TeV_MultiLEP_3Lep_36invfb",
# "ATLAS_13TeV_3b_24invfb",
# "ATLAS_13TeV_3b_discoverySR_24invfb",
# "ATLAS_13TeV_3b_36invfb",
# "ATLAS_13TeV_3b_discoverySR_36invfb"
# "CMS_13TeV_1LEPbb_36invfb",
# "CMS_13TeV_MONOJET_36invfb"
# "CMS_13TeV_2LEPsoft_36invfb",
# "CMS_13TeV_2LEPsoft_36invfb_nocovar",
# "CMS_13TeV_MultiLEP_2SSLep_36invfb",
# "CMS_13TeV_MultiLEP_3Lep_36invfb",
# "CMS_13TeV_2OSLEP_36invfb",
# "CMS_13TeV_2OSLEP_36invfb_nocovar"
# 
# # Skip
# "CMS_13TeV_2LEPsoft_36invfb_nocovar",
# "CMS_13TeV_2OSLEP_36invfb_nocovar",
# "ATLAS_13TeV_3b_discoverySR_24invfb",
# "ATLAS_13TeV_3b_discoverySR_36invfb",
# "ATLAS_13TeV_3b_36invfb",
# "CMS_13TeV_MONOJET_36invfb"
# 
# # Final:
# "ATLAS_13TeV_4LEP_36invfb",
# "ATLAS_13TeV_RJ3L_2Lep2Jets_36invfb",
# "ATLAS_13TeV_RJ3L_3Lep_36invfb",
# "ATLAS_13TeV_3b_24invfb",
# "ATLAS_13TeV_MultiLEP_2Lep0Jets_36invfb",
# "ATLAS_13TeV_MultiLEP_2LepPlusJets_36invfb",
# "ATLAS_13TeV_MultiLEP_3Lep_36invfb",
# "CMS_13TeV_1LEPbb_36invfb",
# "CMS_13TeV_2LEPsoft_36invfb",
# "CMS_13TeV_MultiLEP_2SSLep_36invfb",
# "CMS_13TeV_MultiLEP_3Lep_36invfb",
# "CMS_13TeV_2OSLEP_36invfb",

allow_corr = True # Use the analyses with correlation matrices (much faster to leave them out for testing)
 
analyses = []

a = ColliderAnalysis("ATLAS_13TeV_3b_24invfb")
a.SR_names = ["meff160_ETmiss0__i0", "meff160_ETmiss20__i1", "meff200_ETmiss0__i2", "meff200_ETmiss20__i3", "meff200_ETmiss45__i4", "meff200_ETmiss70__i5", "meff260_ETmiss0__i6", "meff260_ETmiss20__i7", "meff260_ETmiss45__i8", "meff260_ETmiss70__i9", "meff260_ETmiss100__i10", "meff340_ETmiss0__i11", "meff340_ETmiss20__i12", "meff340_ETmiss45__i13", "meff340_ETmiss70__i14", "meff340_ETmiss100__i15", "meff340_ETmiss150__i16", "meff340_ETmiss200__i17", "meff440_ETmiss0__i18", "meff440_ETmiss20__i19", "meff440_ETmiss45__i20", "meff440_ETmiss70__i21", "meff440_ETmiss100__i22", "meff440_ETmiss150__i23", "meff440_ETmiss200__i24", "meff560_ETmiss0__i25", "meff560_ETmiss20__i26", "meff560_ETmiss45__i27", "meff560_ETmiss70__i28", "meff560_ETmiss100__i29", "meff560_ETmiss150__i30", "meff560_ETmiss200__i31", "meff700_ETmiss0__i32", "meff700_ETmiss20__i33", "meff700_ETmiss45__i34", "meff700_ETmiss70__i35", "meff700_ETmiss100__i36", "meff700_ETmiss150__i37", "meff700_ETmiss200__i38", "meff860_ETmiss0__i39", "meff860_ETmiss20__i40", "meff860_ETmiss45__i41", "meff860_ETmiss70__i42", "meff860_ETmiss100__i43", "meff860_ETmiss150__i44", "meff860_ETmiss200__i45", ]
a.SR_n     = [20, 3, 1503, 1137, 65, 0, 1329, 2877, 951, 150, 2, 373, 873, 444, 164, 40, 3, 0, 121, 304, 170, 62, 31, 3, 1, 40, 95, 75, 20, 15, 2, 2, 17, 30, 22, 12, 6, 2, 2, 2, 7, 10, 5, 2, 4, 1, ]
a.SR_b     = [16.21, 0.6503, 1480, 1088, 58.05, 0.2691, 1297, 2860, 991, 149.4, 2.024, 390.1, 884.6, 472.6, 171.1, 36.24, 1.457, 0.006531, 130.3, 310.8, 176.6, 65.1, 22.16, 3.895, 0.4816, 43.46, 102.6, 68.03, 30.72, 14.13, 2.358, 1.08, 13.56, 32.67, 23.78, 12.47, 5.549, 1.728, 0.8551, 2.816, 7.766, 8.968, 4.297, 2.785, 0.9345, 0.4297, ]
a.SR_b_sys = [0.11, 0.0747, 26, 7, 0.39, 0.0547, 8, 36, 6.5, 1, 1.426, 2.6, 13.1, 3, 1.1, 0.24, 0.111, 0.004409, 0.8, 9.5, 1.2, 1.1, 6.03, 0.14, 0.0551, 0.29, 6.6, 0.45, 0.2, 3.19, 1.02, 0.23, 0.09, 3.39, 0.15, 0.08, 0.873, 0.879, 0.1211, 0.246, 2.114, 2.332, 0.335, 0.29, 0.2345, 0.0719, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_13TeV_3b_36invfb")
a.SR_names = ["SR-3b-meff1-A__i0", "SR-3b-meff2-A__i1", "SR-3b-meff3-A__i2", "SR-4b-meff1-A__i3", "SR-4b-meff1-B__i4", "SR-4b-meff2-A__i5", "SR-4b-meff2-B__i6", ]
a.SR_n     = [4, 3, 0, 1, 2, 1, 0, ]
a.SR_b     = [2.5, 2, 0.8, 0.43, 2.6, 0.43, 1.3, ]
a.SR_b_sys = [1, 0.5, 0.5, 0.31, 0.9, 0.27, 0.6, ]
a.N_SR = len(a.SR_names)
#analyses += [a]

a = ColliderAnalysis("ATLAS_13TeV_3b_discoverySR_24invfb")
a.SR_names = ["low-SR-MET0meff440__i0", "low-SR-MET150meff440__i1", ]
a.SR_n     = [1063, 17, ]
a.SR_b     = [1100, 12, ]
a.SR_b_sys = [25, 8, ]
a.N_SR = len(a.SR_names)
analyses += [a] # Use this for NOCORR analyses

a = ColliderAnalysis("ATLAS_13TeV_3b_discoverySR_36invfb")
a.SR_names = ["SR-4b-meff1-A-disc__i0", ]
a.SR_n     = [2, ]
a.SR_b     = [0.7, ]
a.SR_b_sys = [0.5, ]
a.N_SR = len(a.SR_names)
#analyses += [a]

a = ColliderAnalysis("ATLAS_13TeV_4LEP_36invfb")
a.SR_names = ["SR0A__i0", "SR0B__i1", "SR0C__i2", "SR0D__i3", ]
a.SR_n     = [13, 2, 47, 10, ]
a.SR_b     = [10.2, 1.31, 37, 4.1, ]
a.SR_b_sys = [2.1, 0.24, 9, 0.7, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_13TeV_MultiLEP_2Lep0Jets_36invfb")
a.SR_names = ["SR2_SF_loose__i0", "SR2_SF_tight__i1", "SR2_DF_100__i2", "SR2_DF_150__i3", "SR2_DF_200__i4", "SR2_DF_300__i5", ]
a.SR_n     = [153, 9, 78, 11, 6, 2, ]
a.SR_b     = [133, 9.8, 68, 11.5, 2.1, 0.6, ]
a.SR_b_sys = [22, 2.9, 7, 3.1, 1.9, 0.6, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_13TeV_MultiLEP_2LepPlusJets_36invfb")
a.SR_names = ["SR2_int__i0", "SR2_high__i1", "SR2_low__i2", ]
a.SR_n     = [2, 0, 11, ]
a.SR_b     = [4.1, 1.6, 4.2, ]
a.SR_b_sys = [2.6, 1.6, 3.4, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_13TeV_MultiLEP_3Lep_36invfb")
a.SR_names = ["SR3_slep_a__i0", "SR3_slep_b__i1", "SR3_slep_c__i2", "SR3_slep_d__i3", "SR3_slep_e__i4", "SR3_WZ_0Ja__i5", "SR3_WZ_0Jb__i6", "SR3_WZ_0Jc__i7", "SR3_WZ_1Ja__i8", "SR3_WZ_1Jb__i9", "SR3_WZ_1Jc__i10", ]
a.SR_n     = [4, 3, 9, 0, 0, 21, 1, 2, 1, 3, 4, ]
a.SR_b     = [2.2, 2.8, 5.4, 1.4, 1.1, 21.7, 2.7, 1.6, 2.2, 1.8, 1.3, ]
a.SR_b_sys = [0.8, 0.4, 0.9, 0.4, 0.2, 2.9, 0.5, 0.3, 0.5, 0.3, 0.3, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_13TeV_RJ3L_2Lep2Jets_36invfb")
a.SR_names = ["2L2JHIGH__i0", "2L2JINT__i1", "2L2JLOW__i2", "2L2JCOMP__i3", ]
a.SR_n     = [0, 1, 19, 11, ]
a.SR_b     = [1.9, 2.4, 8.4, 2.7, ]
a.SR_b_sys = [0.8, 0.9, 5.8, 2.7, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_13TeV_RJ3L_3Lep_36invfb")
a.SR_names = ["3LHIGH__i0", "3LINT__i1", "3LLOW__i2", "3LCOMP__i3", ]
a.SR_n     = [2, 1, 20, 12, ]
a.SR_b     = [1.1, 2.3, 10, 3.9, ]
a.SR_b_sys = [0.5, 0.5, 2, 1, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("CMS_13TeV_1LEPbb_36invfb")
a.SR_names = ["SRA__i0", "SRB__i1", ]
a.SR_n     = [11, 7, ]
a.SR_b     = [7.5, 8.7, ]
a.SR_b_sys = [2.5, 2.2, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("CMS_13TeV_MONOJET_36invfb")
a.SR_names = ["sr-0__i0", "sr-1__i1", "sr-2__i2", "sr-3__i3", "sr-4__i4", "sr-5__i5", "sr-6__i6", "sr-7__i7", "sr-8__i8", "sr-9__i9", "sr-10__i10", "sr-11__i11", "sr-12__i12", "sr-13__i13", "sr-14__i14", "sr-15__i15", "sr-16__i16", "sr-17__i17", "sr-18__i18", "sr-19__i19", "sr-20__i20", "sr-21__i21", ]
a.SR_n     = [136865, 74340, 42540, 25316, 15653, 10092, 8298, 4906, 2987, 2032, 1514, 926, 557, 316, 233, 172, 101, 65, 46, 26, 31, 29, ]
a.SR_b     = [134500, 73400, 42320, 25490, 15430, 10160, 8480, 4865, 2970, 1915, 1506, 844, 526, 325, 223, 169, 107, 88.1, 52.8, 25, 25.5, 26.9, ]
a.SR_b_sys = [3700, 2000, 810, 490, 310, 170, 140, 95, 49, 33, 32, 18, 14, 12, 9, 8, 6, 5.3, 3.9, 2.5, 2.6, 2.8, ]
a.N_SR = len(a.SR_names)
#analyses += [a]

a = ColliderAnalysis("CMS_13TeV_2LEPsoft_36invfb")
a.SR_names = ["SR1__i0", "SR2__i1", "SR3__i2", "SR4__i3", "SR5__i4", "SR6__i5", "SR7__i6", "SR8__i7", "SR9__i8", "SR10__i9", "SR11__i10", "SR12__i11", ]
a.SR_n     = [2, 15, 19, 18, 1, 0, 3, 1, 2, 1, 2, 0, ]
a.SR_b     = [3.5, 12, 17, 11, 1.6, 3.5, 2, 0.51, 1.4, 1.5, 1.5, 1.2, ]
a.SR_b_sys = [1, 2.3, 2.4, 2, 0.7, 0.9, 0.7, 0.52, 0.7, 0.6, 0.8, 0.6, ]
a.cov = [[1.29, 0.33, 0.45, 0.49, 0.06, 0.09, 0.12, 0.08, 0.12, 0.09, 0.07, 0.12],
 [0.33, 5.09, 1.01, 0.62, 0.12, 0.13,  0.2, 0.12, 0.12, 0.11, 0.15, 0.13],
 [0.45, 1.01, 6.44, 0.78, 0.21, 0.19, 0.18,  0.1, 0.18, 0.18, 0.15, 0.19],
 [0.49, 0.62, 0.78,  3.6, 0.09, 0.07, 0.12, 0.19, 0.19, 0.13, 0.17, 0.32],
 [0.06, 0.12, 0.21, 0.09, 0.59, 0.03, 0.06, 0.03, 0.02, 0.03, 0.03, 0.03],
 [0.09, 0.13, 0.19, 0.07, 0.03, 0.72, 0.03, 0.03, 0.03, 0.04, 0.03, 0.01],
 [0.12,  0.2, 0.18, 0.12, 0.06, 0.03,  0.6, 0.05, 0.04, 0.05, 0.04, 0.05],
 [0.08, 0.12,  0.1, 0.19, 0.03, 0.03, 0.05, 0.17, 0.05, 0.03, 0.04, 0.06],
 [0.12, 0.12, 0.18, 0.19, 0.02, 0.03, 0.04, 0.05, 0.26, 0.05, 0.07, 0.07],
 [0.09, 0.11, 0.18, 0.13, 0.03, 0.04, 0.05, 0.03, 0.05, 0.32, 0.05, 0.04],
 [0.07, 0.15, 0.15, 0.17, 0.03, 0.03, 0.04, 0.04, 0.07, 0.05,  0.2, 0.06],
 [0.12, 0.13, 0.19, 0.32, 0.03, 0.01, 0.05, 0.06, 0.07, 0.04, 0.06, 0.28]]
a.N_SR = len(a.SR_names)
if allow_corr:
    analyses += [a]

a = ColliderAnalysis("CMS_13TeV_2LEPsoft_36invfb_nocovar")
a.SR_names = ["SR1__i0", "SR2__i1", "SR3__i2", "SR4__i3", "SR5__i4", "SR6__i5", "SR7__i6", "SR8__i7", "SR9__i8", "SR10__i9", "SR11__i10", "SR12__i11", ]
a.SR_n     = [2, 15, 19, 18, 1, 0, 3, 1, 2, 1, 2, 0, ]
a.SR_b     = [3.5, 12, 17, 11, 1.6, 3.5, 2, 0.51, 1.4, 1.5, 1.5, 1.2, ]
a.SR_b_sys = [1, 2.3, 2.4, 2, 0.7, 0.9, 0.7, 0.52, 0.7, 0.6, 0.8, 0.6, ]
a.N_SR = len(a.SR_names)
#analyses += [a]

a = ColliderAnalysis("CMS_13TeV_2OSLEP_36invfb")
a.SR_names = ["SR1__i0", "SR2__i1", "SR3__i2", "SR4__i3", "SR5__i4", "SR6__i5", "SR7__i6", ]
a.SR_n     = [57, 29, 2, 0, 9, 5, 1, ]
a.SR_b     = [54.9, 21.6, 6, 2.5, 7.6, 5.6, 1.3, ]
a.SR_b_sys = [7, 5.6, 1.9, 0.9, 2.8, 1.6, 0.4, ]
a.cov = [[52.8, 12.7,    3,  1.2,  4.5,  5.1,  1.2],
 [12.7, 41.4,  3.6,    2,  2.5,    2,  0.7],
 [   3,  3.6,  1.6,  0.6,  0.4,  0.3,  0.1],
 [ 1.2,    2,  0.6,  1.1,  0.3,  0.1,  0.1],
 [ 4.5,  2.5,  0.4,  0.3,  6.5,  1.8,  0.4],
 [ 5.1,    2,  0.3,  0.1,  1.8,  2.4,  0.4],
 [ 1.2,  0.7,  0.1,  0.1,  0.4,  0.4,  0.2]]
a.N_SR = len(a.SR_names)
if allow_corr:
    analyses += [a]

a = ColliderAnalysis("CMS_13TeV_2OSLEP_36invfb_nocovar")
a.SR_names = ["SR1__i0", "SR2__i1", "SR3__i2", "SR4__i3", "SR5__i4", "SR6__i5", "SR7__i6", ]
a.SR_n     = [57, 29, 2, 0, 9, 5, 1, ]
a.SR_b     = [54.9, 21.6, 6, 2.5, 7.6, 5.6, 1.3, ]
a.SR_b_sys = [7, 5.6, 1.9, 0.9, 2.8, 1.6, 0.4, ]
a.N_SR = len(a.SR_names)
#analyses += [a]

a = ColliderAnalysis("CMS_13TeV_MultiLEP_2SSLep_36invfb")
a.SR_names = ["SR1__i0", "SR2__i1", ]
a.SR_n     = [13, 18, ]
a.SR_b     = [12, 18, ]
a.SR_b_sys = [3, 4, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("CMS_13TeV_MultiLEP_3Lep_36invfb")
a.SR_names = ["SR3__i0", "SR4__i1", "SR5__i2", "SR6__i3", "SR7__i4", "SR8__i5", ]
a.SR_n     = [19, 128, 18, 2, 82, 166, ]
a.SR_b     = [19, 142, 22, 1, 109, 197, ]
a.SR_b_sys = [4, 34, 5, 0.6, 28, 42, ]
a.N_SR = len(a.SR_names)
analyses += [a]

# NEW! 8 TeV analyses for paper revison

a = ColliderAnalysis("CMS_8TeV_MultiLEP_3Lep_20invfb")
a.SR_names = ["SR3l_OSSF_mT<120_ETmiss50-100_mll<75__i0", "SR3l_OSSF_mT<120_ETmiss50-100_mll75-105__i1", "SR3l_OSSF_mT<120_ETmiss50-100_mll>105__i2", "SR3l_OSSF_mT<120_ETmiss100-150_mll<75__i3", "SR3l_OSSF_mT<120_ETmiss100-150_mll75-105__i4", "SR3l_OSSF_mT<120_ETmiss100-150_mll>105__i5", "SR3l_OSSF_mT<120_ETmiss150-200_mll<75__i6", "SR3l_OSSF_mT<120_ETmiss150-200_mll75-105__i7", "SR3l_OSSF_mT<120_ETmiss150-200_mll>105__i8", "SR3l_OSSF_mT<120_ETmiss200-250_mll<75__i9", "SR3l_OSSF_mT<120_ETmiss200-250_mll75-105__i10", "SR3l_OSSF_mT<120_ETmiss200-250_mll>105__i11", "SR3l_OSSF_mT120-160_ETmiss50-100_mll<75__i12", "SR3l_OSSF_mT120-160_ETmiss50-100_mll75-105__i13", "SR3l_OSSF_mT120-160_ETmiss50-100_mll>105__i14", "SR3l_OSSF_mT120-160_ETmiss100-150_mll<75__i15", "SR3l_OSSF_mT120-160_ETmiss100-150_mll75-105__i16", "SR3l_OSSF_mT120-160_ETmiss100-150_mll>105__i17", "SR3l_OSSF_mT120-160_ETmiss150-200_mll<75__i18", "SR3l_OSSF_mT120-160_ETmiss150-200_mll75-105__i19", "SR3l_OSSF_mT120-160_ETmiss150-200_mll>105__i20", "SR3l_OSSF_mT120-160_ETmiss200-250_mll<75__i21", "SR3l_OSSF_mT120-160_ETmiss200-250_mll75-105__i22", "SR3l_OSSF_mT120-160_ETmiss200-250_mll>105__i23", "SR3l_OSSF_mT>160_ETmiss50-100_mll<75__i24", "SR3l_OSSF_mT>160_ETmiss50-100_mll75-105__i25", "SR3l_OSSF_mT>160_ETmiss50-100_mll>105__i26", "SR3l_OSSF_mT>160_ETmiss100-150_mll<75__i27", "SR3l_OSSF_mT>160_ETmiss100-150_mll75-105__i28", "SR3l_OSSF_mT>160_ETmiss100-150_mll>105__i29", "SR3l_OSSF_mT>160_ETmiss150-200_mll<75__i30", "SR3l_OSSF_mT>160_ETmiss150-200_mll75-105__i31", "SR3l_OSSF_mT>160_ETmiss150-200_mll>105__i32", "SR3l_OSSF_mT>160_ETmiss200-250_mll<75__i33", "SR3l_OSSF_mT>160_ETmiss200-250_mll75-105__i34", "SR3l_OSSF_mT>160_ETmiss200-250_mll>105__i35", "SR3l_noOSSF_mT<120_ETmiss50-100_mll<100__i36", "SR3l_noOSSF_mT<120_ETmiss50-100_mll>100__i37", "SR3l_noOSSF_mT<120_ETmiss100-150_mll<100__i38", "SR3l_noOSSF_mT<120_ETmiss100-150_mll>100__i39", "SR3l_noOSSF_mT<120_ETmiss150-200_mll<100__i40", "SR3l_noOSSF_mT<120_ETmiss150-200_mll>100__i41", "SR3l_noOSSF_mT<120_ETmiss200-250_mll<100__i42", "SR3l_noOSSF_mT<120_ETmiss200-250_mll>100__i43", "SR3l_noOSSF_mT120-160_ETmiss50-100_mll<100__i44", "SR3l_noOSSF_mT120-160_ETmiss50-100_mll>100__i45", "SR3l_noOSSF_mT120-160_ETmiss100-150_mll<100__i46", "SR3l_noOSSF_mT120-160_ETmiss100-150_mll>100__i47", "SR3l_noOSSF_mT120-160_ETmiss150-200_mll<100__i48", "SR3l_noOSSF_mT120-160_ETmiss150-200_mll>100__i49", "SR3l_noOSSF_mT120-160_ETmiss200-250_mll<100__i50", "SR3l_noOSSF_mT120-160_ETmiss200-250_mll>100__i51", "SR3l_noOSSF_mT>160_ETmiss50-100_mll<100__i52", "SR3l_noOSSF_mT>160_ETmiss50-100_mll>100__i53", "SR3l_noOSSF_mT>160_ETmiss100-150_mll<100__i54", "SR3l_noOSSF_mT>160_ETmiss100-150_mll>100__i55", "SR3l_noOSSF_mT>160_ETmiss150-200_mll<100__i56", "SR3l_noOSSF_mT>160_ETmiss150-200_mll>100__i57", "SR3l_noOSSF_mT>160_ETmiss200-250_mll<100__i58", "SR3l_noOSSF_mT>160_ETmiss200-250_mll>100__i59", "SR3l_SS1tau_mT<120_ETmiss50-100_mll<100__i60", "SR3l_SS1tau_mT<120_ETmiss50-100_mll>100__i61", "SR3l_SS1tau_mT<120_ETmiss100-150_mll<100__i62", "SR3l_SS1tau_mT<120_ETmiss100-150_mll>100__i63", "SR3l_SS1tau_mT<120_ETmiss150-200_mll<100__i64", "SR3l_SS1tau_mT<120_ETmiss150-200_mll>100__i65", "SR3l_SS1tau_mT<120_ETmiss200-250_mll<100__i66", "SR3l_SS1tau_mT<120_ETmiss200-250_mll>100__i67", "SR3l_SS1tau_mT120-160_ETmiss50-100_mll<100__i68", "SR3l_SS1tau_mT120-160_ETmiss50-100_mll>100__i69", "SR3l_SS1tau_mT120-160_ETmiss100-150_mll<100__i70", "SR3l_SS1tau_mT120-160_ETmiss100-150_mll>100__i71", "SR3l_SS1tau_mT120-160_ETmiss150-200_mll<100__i72", "SR3l_SS1tau_mT120-160_ETmiss150-200_mll>100__i73", "SR3l_SS1tau_mT120-160_ETmiss200-250_mll<100__i74", "SR3l_SS1tau_mT120-160_ETmiss200-250_mll>100__i75", "SR3l_SS1tau_mT>160_ETmiss50-100_mll<100__i76", "SR3l_SS1tau_mT>160_ETmiss50-100_mll>100__i77", "SR3l_SS1tau_mT>160_ETmiss100-150_mll<100__i78", "SR3l_SS1tau_mT>160_ETmiss100-150_mll>100__i79", "SR3l_SS1tau_mT>160_ETmiss150-200_mll<100__i80", "SR3l_SS1tau_mT>160_ETmiss150-200_mll>100__i81", "SR3l_SS1tau_mT>160_ETmiss200-250_mll<100__i82", "SR3l_SS1tau_mT>160_ETmiss200-250_mll>100__i83", "SR3l_OS1tau_mT<120_ETmiss50-100_mll<100__i84", "SR3l_OS1tau_mT<120_ETmiss50-100_mll>100__i85", "SR3l_OS1tau_mT<120_ETmiss100-150_mll<100__i86", "SR3l_OS1tau_mT<120_ETmiss100-150_mll>100__i87", "SR3l_OS1tau_mT<120_ETmiss150-200_mll<100__i88", "SR3l_OS1tau_mT<120_ETmiss150-200_mll>100__i89", "SR3l_OS1tau_mT<120_ETmiss200-250_mll<100__i90", "SR3l_OS1tau_mT<120_ETmiss200-250_mll>100__i91", "SR3l_OS1tau_mT120-160_ETmiss50-100_mll<100__i92", "SR3l_OS1tau_mT120-160_ETmiss50-100_mll>100__i93", "SR3l_OS1tau_mT120-160_ETmiss100-150_mll<100__i94", "SR3l_OS1tau_mT120-160_ETmiss100-150_mll>100__i95", "SR3l_OS1tau_mT120-160_ETmiss150-200_mll<100__i96", "SR3l_OS1tau_mT120-160_ETmiss150-200_mll>100__i97", "SR3l_OS1tau_mT120-160_ETmiss200-250_mll<100__i98", "SR3l_OS1tau_mT120-160_ETmiss200-250_mll>100__i99", "SR3l_OS1tau_mT>160_ETmiss50-100_mll<100__i100", "SR3l_OS1tau_mT>160_ETmiss50-100_mll>100__i101", "SR3l_OS1tau_mT>160_ETmiss100-150_mll<100__i102", "SR3l_OS1tau_mT>160_ETmiss100-150_mll>100__i103", "SR3l_OS1tau_mT>160_ETmiss150-200_mll<100__i104", "SR3l_OS1tau_mT>160_ETmiss150-200_mll>100__i105", "SR3l_OS1tau_mT>160_ETmiss200-250_mll<100__i106", "SR3l_OS1tau_mT>160_ETmiss200-250_mll>100__i107", ]
a.SR_n     = [138, 821, 49, 16, 123, 10, 5, 34, 4, 2, 14, 4, 8, 29, 4, 2, 4, 2, 0, 1, 0, 0, 1, 0, 12, 13, 1, 3, 8, 3, 2, 3, 0, 0, 2, 0, 29, 1, 5, 0, 1, 0, 0, 0, 3, 1, 1, 0, 1, 0, 0, 0, 2, 0, 3, 0, 0, 0, 1, 0, 46, 3, 1, 0, 0, 0, 0, 0, 6, 1, 2, 0, 0, 0, 0, 0, 2, 1, 1, 1, 0, 0, 2, 0, 290, 27, 62, 8, 10, 0, 2, 0, 41, 7, 18, 4, 2, 0, 1, 0, 19, 2, 14, 3, 1, 3, 2, 1, ]
a.SR_b     = [132, 776, 45, 20, 131, 10, 4, 34, 2.5, 1.9, 21, 1.2, 9.6, 23, 2.7, 3.3, 3.4, 0.71, 0.26, 0.72, 0.38, 0.29, 0.36, 0.24, 5.8, 7.5, 2.6, 4.5, 4, 1.8, 1.5, 1.5, 0.7, 0.81, 1.1, 0.4, 32, 1.7, 7.3, 0.3, 1, 0.14, 0.53, 0.03, 5.5, 0.25, 1.9, 0.19, 0.46, 0.03, 0.1, 0.008, 3.2, 0.44, 2.1, 0.42, 0.59, 0.1, 0.37, 0.16, 51, 2.8, 6, 0.5, 2, 0.11, 0.9, 0.042, 5.5, 0.35, 0.91, 0.06, 0.15, 0, 0.06, 0.011, 3.1, 0.5, 2.3, 0.4, 0.52, 0.21, 0.41, 0.06, 259, 30, 60, 5.9, 11, 2.3, 2.9, 1.1, 42, 8.3, 17, 2.3, 2, 0.27, 0.8, 0.5, 15, 5.7, 14, 4, 3.7, 1.3, 1.5, 0.7, ]
a.SR_b_sys = [19, 125, 7, 4, 30, 1.9, 0.8, 8, 0.5, 0.4, 7, 0.3, 1.7, 5, 0.5, 0.8, 0.7, 0.22, 0.1, 0.19, 0.14, 0.11, 0.12, 0.2, 1.1, 1.4, 1.2, 1.1, 1, 0.9, 0.4, 0.5, 0.4, 0.21, 0.4, 0.24, 7, 0.4, 1.7, 0.11, 0.3, 0.09, 0.24, 0.03, 1.2, 0.07, 0.5, 0.1, 0.18, 0.03, 0.05, 0.01, 0.8, 0.33, 0.7, 0.19, 0.18, 0.06, 0.13, 0.14, 8, 0.6, 1.3, 0.14, 0.4, 0.07, 0.24, 0.021, 1, 0.13, 0.26, 0.05, 0.1, 0.008, 0.08, 0.012, 0.6, 0.21, 0.5, 0.17, 0.16, 0.11, 0.12, 0.05, 93, 13, 25, 2.6, 5, 1.4, 1.4, 0.6, 16, 2.9, 9, 1.3, 1.2, 0.32, 0.5, 0.4, 8, 2.3, 9, 2.2, 2.1, 1, 1, 0.4, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("CMS_8TeV_MultiLEP_4Lep_20invfb")
a.SR_names = ["SR4l_1OSSF0tau_ETmiss<30__i0", "SR4l_1OSSF0tau_ETmiss30-50__i1", "SR4l_1OSSF0tau_ETmiss50-100__i2", "SR4l_1OSSF0tau_ETmiss>100__i3", "SR4l_1OSSF1tau_ETmiss<30__i4", "SR4l_1OSSF1tau_ETmiss30-50__i5", "SR4l_1OSSF1tau_ETmiss50-100__i6", "SR4l_1OSSF1tau_ETmiss>100__i7", "SR4l_2OSSF0tau_ETmiss<30__i8", "SR4l_2OSSF0tau_ETmiss30-50__i9", "SR4l_2OSSF0tau_ETmiss50-100__i10", "SR4l_2OSSF0tau_ETmiss>100__i11", ]
a.SR_n     = [1, 3, 2, 2, 33, 11, 9, 2, 142, 25, 4, 1, ]
a.SR_b     = [2.3, 1.2, 1.5, 0.8, 25, 11, 9.3, 2.9, 149, 28, 4.5, 0.8, ]
a.SR_b_sys = [0.6, 0.3, 0.4, 0.3, 12, 3.1, 1.9, 0.6, 46, 11, 2.7, 0.3, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_8TeV_1LEPbb_20invfb")
a.SR_names = ["SRA__i0", "SRB__i1", ]
a.SR_n     = [4, 3, ]
a.SR_b     = [5.69, 2.67, ]
a.SR_b_sys = [1.1, 0.69, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_8TeV_2LEPEW_20invfb")
a.SR_names = ["MT2_90_SF__i0", "MT2_90_DF__i1", "MT2_120_SF__i2", "MT2_120_DF__i3", "MT2_150_SF__i4", "MT2_150_DF__i5", "WWa_SF__i6", "WWa_DF__i7", "WWb_SF__i8", "WWb_DF__i9", "WWc_SF__i10", "WWc_DF__i11", "Zjets__i12", ]
a.SR_n     = [33, 21, 5, 5, 3, 2, 73, 70, 26, 17, 10, 11, 1, ]
a.SR_b     = [38.2, 23.3, 8.9, 3.6, 3.2, 1, 86.5, 73.6, 30.2, 18.1, 20.3, 9, 1.4, ]
a.SR_b_sys = [5.1, 3.7, 2.1, 1.2, 0.7, 0.5, 7.4, 7.9, 3.5, 2.6, 3.5, 2.2, 0.6, ]
a.N_SR = len(a.SR_names)
analyses += [a]

a = ColliderAnalysis("ATLAS_8TeV_3LEPEW_20invfb")
a.SR_names = ["SR0tau_a_bin_1__i0", "SR0tau_a_bin_2__i1", "SR0tau_a_bin_3__i2", "SR0tau_a_bin_4__i3", "SR0tau_a_bin_5__i4", "SR0tau_a_bin_6__i5", "SR0tau_a_bin_7__i6", "SR0tau_a_bin_8__i7", "SR0tau_a_bin_9__i8", "SR0tau_a_bin_10__i9", "SR0tau_a_bin_11__i10", "SR0tau_a_bin_12__i11", "SR0tau_a_bin_13__i12", "SR0tau_a_bin_14__i13", "SR0tau_a_bin_15__i14", "SR0tau_a_bin_16__i15", "SR0tau_a_bin_17__i16", "SR0tau_a_bin_18__i17", "SR0tau_a_bin_19__i18", "SR0tau_a_bin_20__i19", "SR1tau__i20", "SR2tau_a__i21", "SR2tau_b__i22", ]
a.SR_n     = [36, 5, 9, 9, 11, 13, 15, 1, 28, 24, 29, 8, 714, 214, 63, 3, 60, 1, 0, 0, 13, 6, 5, ]
a.SR_b     = [23, 4.2, 10.6, 8.5, 12.9, 6.6, 14.1, 1.1, 22.4, 16.4, 27, 5.5, 715, 219, 65, 4.6, 69, 3.4, 1.2, 0.29, 10.3, 6.9, 7.2, ]
a.SR_b_sys = [4, 1.5, 1.8, 1.7, 2.4, 1.9, 2.2, 0.4, 3.6, 2.8, 5, 1.5, 70, 33, 13, 1.7, 9, 1.4, 0.4, 0.18, 1.2, 0.8, 0.8, ]
a.N_SR = len(a.SR_names)
analyses += [a]

# Extra hacking! Set background uncertainties in each analysis to be highly correlated!
#for a in analyses:
#    if a.cov is None:
#        # No covariance matrix: so add one! 0.9 correlation coefficient (1 gives singular matrix)
#        a.cov = [[0.9*a.SR_b_sys[i]*a.SR_b_sys[j] for i in range(a.N_SR)] for j in range(a.N_SR)]
#        for k in range(a.N_SR):
#            a.cov[k][k] = a.SR_b_sys[k]**2 # Need diagonal to be correct variance though
#
#    print(a.cov)

# Dump to YAML
out = {}
for a in analyses:
    out[a.name] = a.as_dict_long_form()
    #print(a.name)
    #print(a.as_dataframe())
    #print()

print("Writing analysis metadata to YAML file")
f = 'CBit_analyses.yaml'
stream = open(f, 'w')
yaml.dump(out, stream)
stream.close()

# Read file back in for checking
stream = open(f, 'r')
analyses_read = collider_analyses_from_long_YAML(stream)
stream.close()

for a in analyses_read:
    print(a.name)
    print(a.as_dataframe())
    print()

#CMS_8TeV_MultiLEP_3Lep_20invfb
#CMS_8TeV_MultiLEP_4Lep_20invfb
#ATLAS_8TeV_1LEPbb_20invfb
#ATLAS_8TeV_2LEPEW_20invfb
#ATLAS_8TeV_3LEPEW_20invfb
