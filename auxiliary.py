
import os
import numpy as np
import pandas as pd
import scipy.io as sio
from one.api import One

bhv_fields_rename = {'trials.included':'include',
                     'trials.visualStim_contrastLeft':'contrastLeft',
                     'trials.visualStim_contrastRight':'contrastRight',
                     'trials.visualStim_times':'stimOn',
                     'trials.response_times':'respTime',
                     'trials.response_choice':'choice',
                     'trials.feedbackType':'feedback'}

def load_justin_betas(path, region='OFC', key='OLS', key_top='ALL_BETA',
                      bottom_key='betas'):
    coefs = sio.loadmat(path)[key_top]
    print(coefs)
    coefs = coefs[key][0, 0][region][0, 0][bottom_key][0, 0]
    print(coefs['arr'])
    return coefs

def split_spks_bhv(ids, ts, beg_ts, end_ts, extra):
    unique_neurs = np.unique(ids)
    spks_cont = np.zeros((len(beg_ts), len(unique_neurs)), dtype=object)
    for i, bt in enumerate(beg_ts):
        bt = np.squeeze(bt)
        et = np.squeeze(end_ts[i])
        mask = np.logical_and(ts > bt - extra, ts < et + extra)
        ids_m, ts_m = ids[mask], ts[mask]
        for j, un in enumerate(unique_neurs):
            mask_n = ids_m == un
            spks_cont[i, j] = ts_m[mask_n]
    return spks_cont, unique_neurs

def load_steinmetz_data(folder, bhv_fields=bhv_fields_rename,
                        trl_start_field='trials.visualStim_times',
                        trl_end_field='trials.feedback_times',
                        max_files=np.inf,
                        spk_times='spikes.times',
                        spike_clusters='spikes.clusters',
                        cluster_channel='clusters.peakChannel',
                        channel_loc='channels.brainLocation',
                        extra=1):
    one = One(cache_dir=folder)
    session_ids = one.search(dataset='trials')
    if max_files is not np.inf:
        session_ids = session_ids[:max_files]
    dates, expers, animals, datas, n_neurs = [], [], [], [], []
    for i, si in enumerate(session_ids):
        bhv_keys = list(bhv_fields.keys())
        data, info = one.load_datasets(si, bhv_keys)
        (trl_start, trl_end), _ = one.load_datasets(si, [trl_start_field,
                                                         trl_end_field])
        
        info_str = info[0]['session_path'].split('/')
        a_i = info_str[2]
        d_i = info_str[3]
        e_i = 'contrast_compare'
        session_dict = {}
        for i, bk in enumerate(bhv_keys):
            session_dict[bhv_fields[bk]] = data[i]
        out = one.load_datasets(si, [spk_times, spike_clusters, cluster_channel,
                                     channel_loc])
        (ts, c_ident, c_channel, chan_loc), _ = out
        spks_cont, unique_neurs = split_spks_bhv(c_ident, ts, trl_start, trl_end,
                                                 extra)
        loc_list = chan_loc['allen_ontology'].to_numpy()
        regions = loc_list[c_channel[unique_neurs].astype(int)]
        session_dict['spikeTimes'] = list(n for n in spks_cont)
        session_dict['neur_regions'] = (regions,)*len(spks_cont)
        df_i = pd.DataFrame.from_dict(session_dict)
        dates.append(d_i)
        expers.append(e_i)
        animals.append(a_i)
        datas.append(df_i)
        n_neurs.append(len(unique_neurs))
    super_dict = dict(date=dates, experiment=expers, animal=animals,
                      data=datas, n_neurs=n_neurs)
    return super_dict

def _add_data_vars(sd, trls, var_names, check=True):
    for i, vn in enumerate(var_names):
        if check:
            all_trls = np.stack(list(t[0, 0][0] for t in trls[:, 0]),
                                axis=0)
            assert np.all(all_trls[0:1] == all_trls)
        vn_vals = trls[0, 0][0, 0][0][:, i]
        sd[vn[0]] = vn_vals
    return sd

def _split_left_right_field(sd, lm, rm, split_fields, new_field,
                            split_keys=('left', 'right')):
    o1f, o2f = split_fields
    f_left = np.zeros(len(sd[o1f]))
    f_right = np.zeros(len(sd[o1f]))
    
    f_left[lm] = sd[o1f][lm]
    f_left[rm] = sd[o2f][rm]
    
    f_right[lm] = sd[o2f][lm]
    f_right[rm] = sd[o1f][rm]
    sd[new_field.format(split_keys[0])] = f_left
    sd[new_field.format(split_keys[1])] = f_right
    return sd

offer1_side_field = 'side of offer 1 (Left = 1 Right =0)'
split_fields = (('prob offer 1', 'prob offer 2'),
                ('rwd offer 1', 'rwd offer 2'),
                ('Expected value offer 1', 'Expected value offer 2'),
                ('Offer 1 on', 'Offer 2 on'),
                ('Offer 1 off', 'Offer 2 off'))
new_fields = ('prob_{}', 'rwd_{}', 'ev_{}', 'offer_{}_on', 'offer_{}_off')

chosen_offer_field = 'choice offer 1 (==1) or 2 (==0)'
def _make_convenience_data_vars(sd, offer1_side=offer1_side_field,
                                split_fields=split_fields,
                                new_fields=new_fields,
                                split_keys=('left','right')):
    o1_left = sd[offer1_side] == 1
    o1_right = np.logical_not(o1_left)
    for i, sf in enumerate(split_fields):
        sd = _split_left_right_field(sd, o1_left, o1_right, sf,
                                     new_fields[i], split_keys=split_keys)
    return sd

def load_fine_data(folder, regions_list=('OFC', 'PCC', 'rOFC'),
                   tv_templ='TrialInfo/{}_Trial_Vars.mat',
                   psth_templ='Neural/{}_psth_rebinned_20ms_original.mat',
                   tv_key='Trial_Vars', psth_key='data',
                   sub_key='subj_ID', vn_key='VarNames',
                   timing_file='event_time_info.mat'):
    dates = []
    animals = []
    datas = []
    n_neurs = []
    regions = []
    psth_timings = []
    timing = sio.loadmat(os.path.join(folder, timing_file))
    psth_time_vec = timing['time_vec_ms'][0]
    event_times = timing['events']
    for j, region in enumerate(regions_list):
        trls_raw = sio.loadmat(os.path.join(folder, tv_templ.format(region)))
        psth_raw = sio.loadmat(os.path.join(folder, psth_templ.format(region)))
        trls = trls_raw[tv_key][sub_key][0, 0]
        psth = psth_raw[psth_key][sub_key][0, 0]
        var_names = trls_raw[tv_key][vn_key][0,0][0]
        animal_names = trls.dtype.names
        for i, an in enumerate(animal_names):
            psth_ij = psth[an][0, 0]
            trls_ij = trls[an][0, 0]
            trl_nums = np.array(list(pij[0]['psth'][0, 0].shape[0]
                                     for pij in psth_ij))
            for k, tn_u in enumerate(np.unique(trl_nums)):
                mask = trl_nums == tn_u
                session_dict = {}
                regions.append(region)
                animals.append(an)
                psth_timings.append(psth_time_vec)
                dates.append('{}-{}-{}'.format(region, an, tn_u))
                
                pop = np.stack(list(pij[0]['psth'][0, 0]
                                    for pij in psth_ij[mask]),
                               axis=1)
                n_neurs.append(pop.shape[1])
                session_dict['neur_regions'] = ((region,)*pop.shape[1],)*pop.shape[0]
                session_dict['psth'] = list(pop)
                session_dict = _add_data_vars(session_dict, trls_ij[mask],
                                              var_names)
                timing_dict = {ename[0]:np.ones(len(pop))*etime[0, 0]
                               for (ename, etime) in event_times}
                session_dict.update(timing_dict)
                session_dict = _make_convenience_data_vars(session_dict)
                session_dict = _make_convenience_data_vars(
                    session_dict, offer1_side=chosen_offer_field,
                    split_keys=('chosen', 'unchosen'))
                
                session_frame = pd.DataFrame.from_dict(session_dict)
                datas.append(session_frame)
    super_dict = dict(date=dates, animal=animals, data=datas, n_neurs=n_neurs,
                      psth_timing=psth_timings)
    return super_dict
