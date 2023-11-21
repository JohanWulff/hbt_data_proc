from typing import List
import numpy as np
import awkward as ak
import vector
import pickle

from helpers import get_num_btag, get_vbf_pair, jet_cat_lookup, adjust_svfit_feats
from helpers import calc_mt, calc_top_masses, get_cvsb_flag, get_jet_quality, get_region, run_htautau


masses = [
    250, 260, 270, 280, 300, 320, 350, 400, 450, 500, 550, 600, 650,
    700, 750, 800, 850, 900, 1000, 1250, 1500, 1750, 2000, 2500, 3000,
]

spins = [0, 2]

cont_htt_inputs= ["met_px", "met_py", 
                  "DeepMET_ResponseTune_px", "DeepMET_ResponseTune_py", "DeepMET_ResolutionTune_px", "DeepMET_ResolutionTune_py",
                  "dphi_l1_l2", "deta_l1_l2",
                  "dau1_px", "dau1_py", "dau1_pz", "dau1_E", "dau1_iso",
                  "dau2_px", "dau2_py", "dau2_pz", "dau2_E", "dau2_iso",
                  "met_cov00", "met_cov01", "met_cov11",
                  "bjet1_px", "bjet1_py", "bjet1_pz", "bjet1_E", "bjet1_bID_deepFlavor", "bjet1_cID_deepFlavor",
                  "bjet2_px", "bjet2_py", "bjet2_pz", "bjet2_E", "bjet2_bID_deepFlavor", "bjet2_cID_deepFlavor"]

cat_htt_inputs = ['pairType', 'dau1_decayMode', 'dau2_decayMode', 'dau1_charge', 'dau2_charge']

id2year = {0: "2016", 1: "2017", 2: "2018", 3: "2016APV"}
year2id = {id2year[i]: i for i in id2year}

spin2id = {'radion': 0, 'graviton': 1}
id2spin = {spin2id[i]: i for i in spin2id}


def fix_sv_feats(events, feats, fix_val = -1):
    sv_fit_conv = events.tauH_SVFIT_mass>0
    for feat in feats:
        np_feat = events[feat].to_numpy()
        np_feat[~sv_fit_conv] = fix_val*np.ones(ak.sum(~sv_fit_conv))
    return events


def calc_feats(events: ak.Array, sample_id: int, sum_w: float, mass: int, spin: int, year: str, add_htautau:bool = False) -> ak.Array:
    events['gen_sample'] = sample_id*np.ones(len(events))
    events['weight'] = np.ones(len(events))
    if int(sum_w) != 1:
        prod = np.ones(len(events)) * events["MC_weight"].to_numpy() * events["prescaleWeight"].to_numpy()* events["L1pref_weight"].to_numpy() * events["PUjetID_SF"].to_numpy()* events["PUReweight"].to_numpy() * events["bTagweightReshape"].to_numpy()* events["trigSF"].to_numpy()* events["IdFakeSF_deep_2d"].to_numpy()
        prod /= sum_w
        events['weight'] = prod

    events = get_vbf_pair(events=events)
    events = get_num_btag(events=events)
    events = jet_cat_lookup(events=events)
    events = get_region(events=events)
    events['split'] = np.zeros(len(events)) 
    np_split = np.asarray(ak.flatten(events.split, axis=0))
    np_split[(events.EventNumber % 4 > 1)] = np.ones(ak.sum((events.EventNumber % 4 > 1)))
    
    sv_fit = vector.array({"pt": events.tauH_SVFIT_pt, "eta": events.tauH_SVFIT_eta,
                            "phi": events.tauH_SVFIT_phi, "mass": events.tauH_SVFIT_mass,}).to_rhophietatau()
    dau1 = vector.array({"pt": events.dau1_pt, "eta": events.dau1_eta,
                        "phi": events.dau1_phi, "e": events.dau1_e,}).to_rhophietatau()
    dau2 = vector.array({"pt": events.dau2_pt, "eta": events.dau2_eta,
                        "phi": events.dau2_phi, "e": events.dau2_e,}).to_rhophietatau()
    b_1 = vector.array({"pt": events.bjet1_pt, "eta": events.bjet1_eta,
                        "phi": events.bjet1_phi, "e": events.bjet1_e,}).to_rhophietatau()
    b_2 = vector.array({"pt": events.bjet2_pt, "eta": events.bjet2_eta,
                            "phi": events.bjet2_phi, "e": events.bjet2_e,}).to_rhophietatau()
    met = vector.array({"pt": events.met_et, "eta": np.zeros(len(events)),
            "phi": events.met_phi, "mass": np.zeros(len(events)), }).to_rhophietatau()
    # continuous feats
    if (int(mass) == 125) & (int(spin) == -1):
        events['res_mass'] = np.random.choice(masses, size=len(events))
        events['gen_target'] = 0*np.ones(len(events))
    else:
        events['res_mass'] = mass*np.ones(len(events))
        events['gen_target'] = np.ones(len(events))
    if int(spin) == -1:
        events['spin'] = np.random.choice(spins, size=len(events))
    else:
        events['spin'] = spin*np.ones(len(events))

    # calculate vector-related feats
    events['bjet1_px'] = b_1.px
    events['bjet1_py'] = b_1.py
    events['bjet1_pz'] = b_1.pz
    events['bjet2_px'] = b_2.px
    events['bjet2_py'] = b_2.py
    events['bjet2_pz'] = b_2.pz
    events['dau1_px'] = dau1.px
    events['dau1_py'] = dau1.py
    events['dau1_pz'] = dau1.pz
    events['dau2_px'] = dau2.px
    events['dau2_py'] = dau2.py
    events['dau2_pz'] = dau2.pz
    events['bjet1_E'] = b_1.E
    events['bjet2_E'] = b_2.E
    events['dau1_E'] = dau1.E
    events['dau2_E'] = dau2.E
    events['met_px'] = met.px
    events['met_py'] = met.py
    events['dR_l1_l2'] = dau1.deltaR(dau2)
    events['dR_b1_b2'] = b_1.deltaR(b_2)
    events['deta_l1_l2'] = np.abs(dau1.deltaeta(dau2))
    events['deta_b1_b2'] = np.abs(b_1.deltaeta(b_2))
    events['dphi_l1_l2'] = np.abs(dau1.deltaphi(dau2))
    events['dR_l1_l2_x_sv_pt'] = events.dR_l1_l2*sv_fit.pt
    events['sv_mass'] = sv_fit.mass
    events['sv_E'] = sv_fit.energy
    events['met_pt'] = met.pt
    events['ll_mt'] = calc_mt(dau1, dau2) 
    # let's not store these 4vectors in events
    h_bb = b_1+b_2
    h_tt_vis = dau1+dau2
    h_tt_met = h_tt_vis+met

    hh = vector.array({"px": h_bb.px+sv_fit.px, "py": h_bb.py+sv_fit.py,
                    "pz": h_bb.pz+sv_fit.pz, "mass": events.HHKin_mass_raw,}).to_rhophietatau()

    h_bb_tt_met_kinfit = vector.array({"px": h_bb.px+h_tt_met.px, "py": h_bb.py+h_tt_met.py,
                                "pz": h_bb.pz+h_tt_met.pz, "mass": events.HHKin_mass_raw}).to_rhophietatau()
    hh_kinfit_chi2 = events.HHKin_mass_raw_chi2.to_numpy()
    inf_mask = (hh_kinfit_chi2 == np.inf) | (hh_kinfit_chi2 == -np.inf)
    hh_kinfit_chi2[inf_mask] = -1*np.ones(ak.sum(inf_mask)) 
    hh_kinfit_conv = hh_kinfit_chi2>0
    hh_kinfit_chi2[~hh_kinfit_conv] = -1*np.ones(ak.sum(~hh_kinfit_conv)) 
    sv_fit_conv = events.tauH_SVFIT_mass>0
    hh[np.logical_and(~hh_kinfit_conv, sv_fit_conv)] = (h_bb+sv_fit)[np.logical_and(~hh_kinfit_conv, sv_fit_conv)]
    hh[np.logical_and(~hh_kinfit_conv, ~sv_fit_conv)] = (h_bb+h_tt_met)[np.logical_and(~hh_kinfit_conv, ~sv_fit_conv)]
    hh[np.logical_and(hh_kinfit_conv, ~sv_fit_conv)] = h_bb_tt_met_kinfit[np.logical_and(hh_kinfit_conv, ~sv_fit_conv)]

    events = fix_sv_feats(events, feats=['sv_mass', 'sv_E', 'dR_l1_l2_x_sv_pt'])
    events['hh_pt'] = hh.pt
    events['h_bb_mass'] = h_bb.m
    events['diH_mass_met'] = (h_bb+h_tt_met).M
    events['deta_hbb_httvis'] = np.abs(h_bb.deltaeta(h_tt_vis))
    events['dphi_hbb_met'] = np.abs(h_bb.deltaphi(met))

    # set sv_mt to -1 as default
    events['sv_mt'] = -1*np.ones(len(events))
    # calculate sv_mt only for events where sv_fit converged (mass > 0)
    np_sv_mt = np.asarray(ak.flatten(events.sv_mt, axis=0))
    np_sv_mass = np.asarray(ak.flatten(events.sv_mass, axis=0))
    np_sv_mt[np_sv_mass > 0] = calc_mt(v=sv_fit[np_sv_mass > 0], met=met[np_sv_mass > 0])

    events['dau1_mt'] = calc_mt(v=dau1, met=met)
    events['dau2_mt'] = calc_mt(v=dau2, met=met)
    events['dau1_charge'] = events['dau1_flav']/np.abs(events['dau1_flav'])
    events['dau2_charge'] = events['dau2_flav']/np.abs(events['dau2_flav'])

    top_masses, top_mass_idx = calc_top_masses(l_1=dau1, l_2=dau2, b_1=b_1, b_2=b_2, met=met)
    events['top_1_mass'] = np.array([i[0] for i in top_masses], dtype='float32')
    events['top_2_mass'] = np.array([i[1] for i in top_masses], dtype='float32')
    events['top_mass_idx'] = np.asarray(top_mass_idx, dtype="int32")
    events['year'] = year2id[year]*np.ones(len(events))
    events['boosted'] = events['isBoosted']
    events['is_vbf'] = events['has_vbf_pair']

    events['strat_key'] = ak.values_astype((1**events.gen_sample)*
                                          (3**events.jet_cat)*
                                          (5**events.pairType)*
                                          (7**events.year)*
                                          (11**events.region), np.int64)
    events = adjust_svfit_feats(events=events)
    events = get_jet_quality(events, int(year.rstrip("APV")))
    # add htautau inputs
    if add_htautau:
        events = run_htautau(events, cont_htt_inputs, cat_htt_inputs,) 
    return events