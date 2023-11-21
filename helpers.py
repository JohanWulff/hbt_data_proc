import awkward as ak
from typing import List

import torch
import vector
import numpy as np

from htautau import HTauTauRegression

# https://github.com/GilesStrong/cms_hh_proc_interface/blob/master/processing/interface/feat_comp.hh lines 36-38
deep_bjet_wps = {2016: [0,0.0614,0.3093,0.7221],
                 2017: [0,0.0521,0.3033,0.7489],
                 2018: [0,0.0494,0.2770,0.7264]}

# treated the same for each year here: 
# https://github.com/GilesStrong/cms_hh_proc_interface/blob/master/processing/interface/feat_comp.hh lines 39-44
cvsl_wps = [0,0.03,0.085,0.48]
# note descending order 
cvsb_wps = [0.4,0.29,0.05,0]


def struct_to_float_array(arr):
    return arr.astype([(name, np.float32) for name in arr.dtype.names], copy=False).view(np.float32).reshape((-1, len(arr.dtype)))


def run_htautau(events: ak.Array, cont_htt_inputs: List, cat_htt_inputs: List):
    set_0_mask = events.EventNumber % 4 < 2
    set_1_mask = events.EventNumber % 4 > 1
    set_0 = events[set_0_mask]
    set_1 = events[set_1_mask]

    x_cont_tau_set_0 = torch.from_numpy(np.asarray(struct_to_float_array(ak.to_numpy(set_0[cont_htt_inputs])), dtype=np.float32))
    x_cont_tau_set_1 = torch.from_numpy(np.asarray(struct_to_float_array(ak.to_numpy(set_1[cont_htt_inputs])), dtype=np.float32))
    x_cat_tau_set_0 = torch.from_numpy(np.asarray(struct_to_float_array(ak.to_numpy(set_0[cat_htt_inputs])), dtype=np.int32))
    x_cat_tau_set_1 = torch.from_numpy(np.asarray(struct_to_float_array(ak.to_numpy(set_1[cat_htt_inputs])), dtype=np.int32))

    tau_model_0 = HTauTauRegression(model_file='/eos/user/j/jowulff/res_HH/hbt_resonant_evaluation/htt_v3_set_0.pkl',)
    tau_model_1 = HTauTauRegression(model_file='/eos/user/j/jowulff/res_HH/hbt_resonant_evaluation/htt_v3_set_1.pkl',)
    tau_model_0.eval()
    tau_model_1.eval()
    with torch.no_grad():
        tau_out_cls_0, tau_out_reg_0, _, tau_out_cstm_0, _= tau_model_1(x_cont_tau_set_0, x_cat_tau_set_0)
        tau_out_cls_1, tau_out_reg_1, _, tau_out_cstm_1, _= tau_model_0(x_cont_tau_set_1, x_cat_tau_set_1)

    tau_out_cls_0 = tau_out_cls_0.numpy()
    tau_out_cls_1 = tau_out_cls_1.numpy()
    tau_out_reg_0 = tau_out_reg_0.numpy()
    tau_out_reg_1 = tau_out_reg_1.numpy()
    tau_out_cstm_0 = tau_out_cstm_0.numpy()
    tau_out_cstm_1 = tau_out_cstm_1.numpy()

    tau_out_cls = -5*np.ones(shape=(len(events), tau_out_cls_0.shape[-1]))
    tau_out_reg = -5*np.ones(shape=(len(events), tau_out_reg_0.shape[-1]))
    tau_out_cstm = -5*np.ones(shape=(len(events), tau_out_cstm_0.shape[-1]))

    tau_out_cls[set_0_mask] = tau_out_cls_0
    tau_out_cls[set_1_mask] = tau_out_cls_1

    tau_out_reg[set_0_mask] = tau_out_reg_0
    tau_out_reg[set_1_mask] = tau_out_reg_1

    tau_out_cstm[set_0_mask] = tau_out_cstm_0
    tau_out_cstm[set_1_mask] = tau_out_cstm_1

    for i in range(tau_out_cls.shape[-1]):
        events[f"tautau_cls_{i}"] = tau_out_cls[:, i]
    for i in range(tau_out_reg.shape[-1]):
        events[f"tautau_reg_{i}"] = tau_out_reg[:, i]
    for i in range(tau_out_cstm.shape[-1]):
        events[f"tautau_cstm_{i}"] = tau_out_cstm[:, i]
    return events


def get_num_btag(events: ak.Array) -> ak.Array:
    num_btag_Loose_1 = np.logical_or((events.bjet1_bID_deepFlavor > 0.049),(events.bjet2_bID_deepFlavor > 0.049))
    num_btag_Loose_2 = (events.bjet1_bID_deepFlavor > 0.049) &  (events.bjet2_bID_deepFlavor > 0.049)
    num_btag_Medium_1 = np.logical_or((events.bjet1_bID_deepFlavor > 0.2783),(events.bjet2_bID_deepFlavor > 0.2783))
    num_btag_Medium_2 = (events.bjet1_bID_deepFlavor > 0.2783) &  (events.bjet2_bID_deepFlavor > 0.2783)
    # create new col and convert mask to int
    events['num_btag_Loose'] = num_btag_Loose_1*1
    # get np values as a view of the ak col
    num_btag_Loose = np.asarray(ak.flatten(events.num_btag_Loose, axis=0))
    num_btag_Loose[num_btag_Loose_2] = 2*np.ones(len(num_btag_Loose_2[num_btag_Loose_2==True])) 
    # same for medium
    events['num_btag_Medium'] = num_btag_Medium_1*1
    num_btag_Medium = np.asarray(ak.flatten(events.num_btag_Medium, axis=0))
    num_btag_Medium[num_btag_Medium_2] = 2*np.ones(len(num_btag_Medium_2[num_btag_Medium_2==True])) 
    return events


def get_vbf_pair(events: ak.Array) -> ak.Array:
    vbf_cond_1 = ( events.isVBF == 1 ) & ( events.VBFjj_mass > 500 ) & ( events.VBFjj_deltaEta > 3 )
    vbf_cond_2= (( events.dau1_pt > 25 ) & ( events.dau2_pt > 25 ) & np.logical_or((events.dau1_pt <= 40 ),( events.dau2_pt <= 40 )))
    vbf_cond_3 = (( events.VBFjj_mass > 800 ) & ( events.VBFjet1_pt > 140 ) & ( events.VBFjet2_pt > 60 ))

    vbf_mask = vbf_cond_1 & np.logical_or((vbf_cond_2) & (vbf_cond_3),(events.isVBFtrigger == 0))
    events['has_vbf_pair'] = vbf_mask
    return events


def jet_cat_lookup(events: ak.Array) -> ak.Array:
    vbf_mask = (( events.has_vbf_pair ) & ( events.num_btag_Loose >= 1 )) # 2j1b+_VBFL, 2j1b+_VBF, 2j1b+_VBFT
    # isBoosted is an int
    no_vbf_2j2bL = (( ~events.has_vbf_pair ) & ( events.isBoosted ).to_numpy().astype(bool) & ( events.num_btag_Loose >= 2 ))
    no_vbf_2j2bR = (( ~events.has_vbf_pair ) & ( events.num_btag_Medium >= 2 ))
    no_vbf_2j1bR = (( ~events.has_vbf_pair ) & ( events.num_btag_Loose >= 1 ))
    no_vbf_2j0bR = (( ~events.has_vbf_pair ) & ( events.num_btag_Loose == 0 ))
    events['jet_cat'] = vbf_mask*1
    jet_cat_np = np.asarray(ak.flatten(events.jet_cat, axis=0))
    jet_cat_np[no_vbf_2j0bR] = 2*np.ones(len(no_vbf_2j0bR[no_vbf_2j0bR==1]))
    jet_cat_np[no_vbf_2j1bR] = 3*np.ones(len(no_vbf_2j1bR[no_vbf_2j1bR==1]))
    jet_cat_np[no_vbf_2j2bR] = 4*np.ones(len(no_vbf_2j2bR[no_vbf_2j2bR==1]))
    jet_cat_np[no_vbf_2j2bL] = 5*np.ones(len(no_vbf_2j2bL[no_vbf_2j2bL==1]))
    return events


def calc_top_masses(l_1, l_2, b_1, b_2, met):
    vector_mass_top = np.array([
        ((l_1 + b_1 + met).mass, (l_2 + b_2).mass),
        ((l_1 + b_2 + met).mass, (l_2 + b_1).mass),
        ((l_1 + b_1).mass, (l_2 + b_2 + met).mass),
        ((l_1 + b_2).mass, (l_2 + b_1 + met).mass)
    ])
    
    distance = np.array([(mass[0] - 172.5) ** 2 + (mass[1] - 172.5) ** 2 for mass in vector_mass_top])
    min_dis = np.argmin(distance, axis=0)
    top_masses = [(vector_mass_top[m][0][i], vector_mass_top[m][1][i]) for i,m in enumerate(min_dis)]
    top_masses = [sorted(m, reverse=True) for m in top_masses]
    return top_masses, min_dis


def get_cvsb_flag(score:ak.Array):
    cvsb_flag = np.zeros(len(score))
    np_score = score.to_numpy()
    for wp, tag in zip(cvsb_wps, [1,2,3,4]):
        cvsb_flag[np_score>=wp] = tag*np.ones(len(cvsb_flag[np_score>=wp]))
    return cvsb_flag


def adjust_svfit_feats(events: ak.Array, fixlist=['dR_l1_l2_x_sv_pt', 'sv_mass', 'sv_E', 'sv_mt']):
    sv_fit_conv = events.tauH_SVFIT_mass>0
    np_fix_cols = events[fixlist].to_numpy()
    np_fix_cols[~sv_fit_conv] = np.empty(ak.sum((~sv_fit_conv))).fill(np.nan)
    return events


def adjust_hh_kinfit(events:ak.Array)->ak.Array:
    hh_kinfit_chi2 = events.HHKin_mass_raw_chi2.to_numpy()
    # get a kinfit convergence col
    hh_kinfit_conv = (hh_kinfit_chi2>0).to_numpy()
    hh_kinfit_chi2[~hh_kinfit_conv] = np.empty(len(hh_kinfit_chi2[~hh_kinfit_conv])).fill(np.nan)
    return events


def get_jet_quality(events:ak.Array, year: int):
    for jet_idx in [1, 2]:
        jet_quality = np.zeros(len(events))
        np_jet_score = events[f'bjet{jet_idx}_bID_deepFlavor'].to_numpy()
        for wp, tag in zip(deep_bjet_wps[year], [1,2,3,4]):
            jet_quality[np_jet_score>=wp] = tag*np.ones(len(np_jet_score[np_jet_score>=wp]))
        events[f'jet_{jet_idx}_quality'] = jet_quality
    return events


def calc_mt(v: vector, met: vector) -> np.array:
    # taken from https://github.com/GilesStrong/cms_hh_proc_interface/blob/master/processing/src/feat_comp.cc line 219
    mt = np.sqrt(2*v.pt*met.pt * (1-np.cos(v.deltaphi(met))))
    # set non-finite (probably due to kinfit or svift non-convergence) to -1
    mt[~np.isfinite(mt)] = -1*np.ones(np.sum(~np.isfinite(mt)))
    return mt


def get_jet_cat(events: ak.Array) -> ak.Array:
    events['jet_cat'] = np.zeros(len(events))
    jc_five_mask = (events.is_vbf) & (events.num_btag_loose>=1)
    jc_four_mask = (~events.is_vbf) & (events.boosted) & (events.num_btag_loose>=2)
    jc_three_mask = (~events.is_vbf) & (events.num_btag_medium>=2)
    jc_two_mask = (~events.is_vbf) & (events.num_btag_loose>=1)
    jc_one_mask = (~events.is_vbf) & (events.num_btag_loose==0)
    events[jc_one_mask, 'jet_cat'] = 1*np.ones(np.sum(jc_one_mask))
    events[jc_two_mask, 'jet_cat'] = 2*np.ones(np.sum(jc_two_mask))
    events[jc_three_mask, 'jet_cat'] = 3*np.ones(np.sum(jc_three_mask))
    events[jc_four_mask, 'jet_cat'] = 4*np.ones(np.sum(jc_four_mask))
    events[jc_five_mask, 'jet_cat'] = 5*np.ones(np.sum(jc_five_mask))
    return events


def get_region(events: ak.Array) -> ak.Array:

    sr_et = ( events.isOS != 0 ) & ( events.pairType==1 ) & ( events.dau1_eleMVAiso == 1 ) & ( events.dau2_deepTauVsJet >= 5 )  # A
    b_et = ( events.isOS == 0 ) & ( events.pairType==1 ) & ( events.dau1_eleMVAiso == 1 ) & ( events.dau2_deepTauVsJet >= 5 ) # B
    c_et = ( events.isOS != 0 ) & ( events.pairType==1 ) & ( events.dau1_eleMVAiso == 1 ) & ( events.dau2_deepTauVsJet >= 1 ) & ( events.dau2_deepTauVsJet < 5 ) # C
    d_et = ( events.isOS == 0 ) & ( events.pairType==1 ) & ( events.dau1_eleMVAiso == 1 ) & ( events.dau2_deepTauVsJet >= 1 ) & ( events.dau2_deepTauVsJet < 5 ) # D

    sr_mt = ( events.isOS != 0 ) & ( events.pairType==2 ) & ( events.dau1_deepTauVsJet >= 5 ) & ( events.dau2_deepTauVsJet >= 5 )  # A
    b_mt = ( events.isOS == 0 ) & ( events.pairType==2 ) & ( events.dau1_deepTauVsJet >= 5 ) & ( events.dau2_deepTauVsJet >= 5 ) # B
    c_mt = ( events.isOS != 0 ) & ( events.pairType==2 ) & ( events.dau1_deepTauVsJet >= 5 ) & ( events.dau2_deepTauVsJet >= 1 ) & ( events.dau2_deepTauVsJet < 5 ) # C
    d_mt = ( events.isOS == 0 ) & ( events.pairType==2 ) & ( events.dau1_deepTauVsJet >= 5 ) & ( events.dau2_deepTauVsJet >= 1 ) & ( events.dau2_deepTauVsJet < 5 ) # D

    sr_tt = ( events.isOS != 0 ) & ( events.pairType==0 ) & ( events.dau1_iso < 0.15 ) & ( events.dau2_deepTauVsJet >= 5 )  # A
    b_tt = ( events.isOS == 0 ) & ( events.pairType==0 ) & ( events.dau1_iso < 0.15 ) & ( events.dau2_deepTauVsJet >= 5 ) # B
    c_tt = ( events.isOS != 0 ) & ( events.pairType==0 ) & ( events.dau1_iso < 0.15 ) & ( events.dau2_deepTauVsJet >= 1 ) & ( events.dau2_deepTauVsJet < 5 ) # C
    d_tt = ( events.isOS == 0 ) & ( events.pairType==0 ) & ( events.dau1_iso < 0.15 ) & ( events.dau2_deepTauVsJet >= 1 ) & ( events.dau2_deepTauVsJet < 5 ) # D
    # create new col and set all events in the SR to 1
    sr_mask = (sr_et) | (sr_mt) | (sr_tt)
    b_mask = (b_et) | (b_mt) | (b_tt)
    c_mask = (c_et) | (c_mt) | (c_tt)
    d_mask = (d_et) | (d_mt) | (d_tt)
    events['region'] = 0*np.ones(len(events))
    region_np = np.asarray(ak.flatten(events.region, axis=0))
    ## b -> 2 , c -> 3, d -> 4
    region_np[sr_mask] = np.ones(np.sum(sr_mask))
    region_np[b_mask] = 2*np.ones(np.sum(b_mask))
    region_np[c_mask] = 3*np.ones(np.sum(c_mask))
    region_np[d_mask] = 4*np.ones(np.sum(d_mask))
    return events
