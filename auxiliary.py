
import os
import numpy as np
import pandas as pd
import scipy.io as sio
import re
import pickle

import arviz as az
# from one.api import One

import general.utility as u

bhv_fields_rename = {'trials.included':'include',
                     'trials.visualStim_contrastLeft':'contrastLeft',
                     'trials.visualStim_contrastRight':'contrastRight',
                     'trials.visualStim_times':'stimOn',
                     'trials.response_times':'respTime',
                     'trials.response_choice':'choice',
                     'trials.feedbackType':'feedback'}
monkey_list = np.array(
    ['Batman', 'Calvin', 'Hobbes', 'Pumbaa', 'Spock', 'Vader']
)
region_monkey_dict = {
    "OFC": ('Pumbaa', 'Spock', 'Vader'),
    "PCC": ('Pumbaa', 'Spock'),
    "VS": ('Batman', 'Calvin'),
    "pgACC": ('Batman', 'Vader'),
    "vmPFC": ('Batman', 'Hobbes'),
    "all": ('Batman', 'Calvin', 'Hobbes', 'Pumbaa', 'Spock', 'Vader')
}


def filter_low_performance(data, perf_thr, filter_len=50):
    corrs = data["subj_ev_chosen"] - data["subj_ev_unchosen"] > 0
    trl_nums = data["trial_num"]

    session_mask = np.zeros(len(trl_nums), dtype=bool)
    for i, trl in enumerate(trl_nums):
        corr = corrs[i]
        filt = np.ones(filter_len)/filter_len
        corr_filt = np.convolve(corr, filt, mode="valid")
        
        mask = np.all(corr_filt >= perf_thr)
        session_mask[i] = mask
    return data.session_mask(session_mask)


def load_monkey_region_runs(
        *runinds,
        folder="multiple_representations/decoding_cluster/",
        template="decoding_m-(?P<monkey>[a-zA-Z]+)_(r-)?[A-Za-z]+_{runind}\.pkl",
):
    ris = "({})".format("|".join(runinds))
    pattern = template.format(runind=ris)
    out_dict = {}
    for i, (_, gd, data) in enumerate(u.load_folder_regex_generator(folder, pattern)):
        m_dict = out_dict.get(gd["monkey"], {})
        for k, v in data.items():
            k_dict = m_dict.get(k, {})
            k_dict.update(v)
            m_dict[k] = k_dict
        out_dict[gd["monkey"]] = m_dict
    return out_dict
    


def load_monkey_decoding_runs(
        runind,
        folder="multiple_representations/decoding_cluster/",
        template="decoding_m-(?P<monkey>[a-zA-Z]+)_(r-)?[A-Za-z]+_{runind}\.pkl",        
):
    pattern = template.format(runind=runind)
    out_dict = {}
    for i, (_, gd, data) in enumerate(u.load_folder_regex_generator(folder, pattern)):
        out_dict[gd["monkey"]] = data 
    return out_dict


def make_decoding_run_db(
        folder="multiple_representations/decoding_cluster/",
        template="decoding_(m-[a-zA-Z]+_)?(r-)?[A-Za-z]+_{runind}\.pkl",
        runind_pattern="[0-9]+",
):
    pattern = template.format(runind=runind_pattern)
    keys_all = []
    args_all = []
    for i, (_, _, data) in enumerate(u.load_folder_regex_generator(folder, pattern)):
        args = data["args"]
        keys_all.extend(list(args.keys()))
        args_all.append(args)
    uk = np.unique(keys_all)
    full_dict = {}
    for arg in args_all:
        for k in uk:
            k_list = full_dict.get(k, [])
            k_list.append(arg.get(k))
            full_dict[k] = k_list
    return pd.DataFrame(full_dict)
    


def load_decoding_runs(
        *runinds,
        folder="multiple_representations/decoding_cluster/",
        template="decoding_(m-[a-zA-Z]+_)?(r-)?[A-Za-z]+_{runind}\.pkl",
):
    runind_comb = "|".join(list(str(ri) for ri in runinds))
    runind_re = "({})".format(runind_comb)
    pattern = template.format(runind=runind_re)
    for i, (_, _, data) in enumerate(u.load_folder_regex_generator(folder, pattern)):
        if i == 0:
            comb_dict = {k: {} for k in data.keys()}
        for k, v_dict in data.items():
            if k == "args" or len(comb_dict[k]) == 0:
                comb_dict[k].update(v_dict)
            else:
                for r, res in v_dict.items():
                    curr = comb_dict[k].get(r)
                    if curr is None:
                        comb_dict[k][r] = res
                    else:
                        for cond, out in res.items():
                            if len(out) == 3:
                                d_full, xs, g_full = comb_dict[k][r].get(cond)
                                d_full = np.concatenate(
                                    (d_full, out[0]), axis=0
                                )
                                g_full = np.concatenate(
                                    (g_full, out[2]), axis=0
                                )
                                comb_dict[k][r][cond] = (d_full, xs, g_full)
                            elif len(out) == 7:
                                new_out = {
                                    quant: np.concatenate((q_full, out[quant]), axis=0)
                                    for quant, q_full in out.items()
                                }
                                comb_dict[k][r][cond] = new_out
    return comb_dict


def remove_pops(dec_dict):
    new_dict = {}
    for r, suff_dict in dec_dict.items():
        new_dict[r] = {}
        for suff, dec_out in suff_dict.items():
            dec, xs, _, _, _, _, gen = dec_out
            new_dict[r][suff] = (dec, xs, gen)
    return new_dict


def save_model_fits(fit_dict, output_folder):
    os.mkdir(output_folder)
    for key, item in fit_dict.items():
        for i, fit in enumerate(item[1]):
            name = 'azf_{}.nc'.format(i)
            new_path = os.path.join(output_folder, name)            
            fit.to_netcdf(new_path)
            item[1][i] = name
    d_path = os.path.join(output_folder, 'fit_dict.pkl')
    pickle.dump(fit_dict, open(d_path, 'wb'))
    return fit_dict

def load_model_fits(folder, load_if_region=None):
    fd_path = os.path.join(folder, 'fit_dict.pkl')
    fd = pickle.load(open(fd_path, 'rb'))
    for key, item in fd.items():
        for i, fit_path in enumerate(item[1]):
            if ((load_if_region is not None and key[0][i] == load_if_region)
                or load_if_region is None):
                _, fit_name = os.path.split(fit_path)
                fit_targ = os.path.join(folder, fit_name)
                fit = az.from_netcdf(fit_targ)
                item[1][i] = fit
    return fd


all_types = ('noise', 'null', 'null_spline', 'interaction',
             'interaction_spline')
all_nums = range(289 + 1)
# all_nums = range(50 + 1)
def load_region_type_fits(folder, type_, region,
                          template='fit_{type}_{num}',
                          nums=all_nums):
    out_dict = {}
    out_data = {}
    for num in nums:
        n_dict = {}
        path = os.path.join(folder, template.format(num=num, type=type_))
        out_nt = load_model_fits(path, load_if_region=region)
        for k, sess in out_nt.items():
            (regions, animal, info) = k            
            for i in range(len(sess[1])):
                if regions[i] == region:
                    out_dict[(regions[i], animal, info, i)] = out_nt[k][1][i]
                    out_data[(regions[i], animal, info, i)] = sess[0]
    return out_dict, out_data

def make_fit_matrix(fit_dict, coeff_key='beta', intercept_key='alpha'):
    rows = []
    first_eg = list(fit_dict.values())[0].posterior[coeff_key]
    n_samps, dim = np.concatenate(first_eg, axis=0).shape
    out_mat = np.zeros((n_samps, len(fit_dict), dim))
    out_inter = np.zeros((n_samps, len(fit_dict), 1))
    for i, (k, v) in enumerate(fit_dict.items()):
        out_mat[:, i] = np.concatenate(v.posterior[coeff_key], axis=0)
        out_inter[:, i, 0] = np.concatenate(v.posterior[intercept_key], axis=0)
    return out_mat, out_inter    

def load_many_fits(folder, types=all_types, nums=all_nums,
                   template='fit_{type}_{num}', thr=0):
    az_dict = {}
    comp_dict = {}
    for num in nums:
        n_dict = {}
        for type_ in types:
            path = os.path.join(folder, template.format(num=num, type=type_))
            out_nt = load_model_fits(path)
            n_dict[type_] = out_nt
        for k, sess in out_nt.items():
            (regions, animal, info) = k 
            for i in range(len(sess[1])):
                neur_type_dict = {t:n_dict[t][k][1][i] for t in types}
                x = az.compare(neur_type_dict)
                comp_dict[(regions[i], animal, info, i)] = x
                az_dict[(regions[i], animal, info, i)] = neur_type_dict
    return az_dict, comp_dict

def resave_mats(folder, mat_templ='.*\.mat'):
    fls = os.listdir(folder)
    for fl in fls:
        m = re.match(mat_templ, fl)
        if m is not None:
            new_data = {}
            data = sio.loadmat(os.path.join(folder, fl))
            for k in data.keys():
                if k[0] != '_':
                    names = data[k].dtype.names
                    if names is not None:
                        new_names = list(n.replace(' ', '_')
                                         for n in names)
                        new_names = list(n.replace('-', 'X')
                                         for n in names)
                        data[k].dtype.names = new_names
                    new_data[k] = data[k]
            try:
                sio.savemat(os.path.join(folder, 're_' + fl), new_data)
            except:
                print(new_data)

def accumulate_time(pop, keepdim=True, ax=1):
    out = np.concatenate(list(pop[..., i] for i in range(pop.shape[-1])),
                         axis=ax)
    if keepdim:
        out = np.expand_dims(out, -1)
    return out

def load_justin_betas(path, region='OFC', key='OLS', key_top='ALL_BETA',
                      bottom_key='betas'):
    coefs = sio.loadmat(path)[key_top]
    coefs = coefs[key][0, 0][region][0, 0][bottom_key][0, 0]
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

# def load_steinmetz_data(folder, bhv_fields=bhv_fields_rename,
#                         trl_start_field='trials.visualStim_times',
#                         trl_end_field='trials.feedback_times',
#                         max_files=np.inf,
#                         spk_times='spikes.times',
#                         spike_clusters='spikes.clusters',
#                         cluster_channel='clusters.peakChannel',
#                         channel_loc='channels.brainLocation',
#                         extra=1):
#     one = One(cache_dir=folder)
#     session_ids = one.search(dataset='trials')
#     if max_files is not np.inf:
#         session_ids = session_ids[:max_files]
#     dates, expers, animals, datas, n_neurs = [], [], [], [], []
#     for i, si in enumerate(session_ids):
#         bhv_keys = list(bhv_fields.keys())
#         data, info = one.load_datasets(si, bhv_keys)
#         (trl_start, trl_end), _ = one.load_datasets(si, [trl_start_field,
#                                                          trl_end_field])
        
#         info_str = info[0]['session_path'].split('/')
#         a_i = info_str[2]
#         d_i = info_str[3]
#         e_i = 'contrast_compare'
#         session_dict = {}
#         for i, bk in enumerate(bhv_keys):
#             session_dict[bhv_fields[bk]] = data[i]
#         out = one.load_datasets(si, [spk_times, spike_clusters, cluster_channel,
#                                      channel_loc])
#         (ts, c_ident, c_channel, chan_loc), _ = out
#         spks_cont, unique_neurs = split_spks_bhv(c_ident, ts, trl_start, trl_end,
#                                                  extra)
#         loc_list = chan_loc['allen_ontology'].to_numpy()
#         regions = loc_list[c_channel[unique_neurs].astype(int) - 1]
#         session_dict['spikeTimes'] = list(n for n in spks_cont)
#         session_dict['neur_regions'] = (regions,)*len(spks_cont)
#         df_i = pd.DataFrame.from_dict(session_dict)
#         dates.append(d_i)
#         expers.append(e_i)
#         animals.append(a_i)
#         datas.append(df_i)
#         n_neurs.append(len(unique_neurs))
#     super_dict = dict(date=dates, experiment=expers, animal=animals,
#                       data=datas, n_neurs=n_neurs)
#     return super_dict

rwd_conversion = {0.15:1., 0.18:2., 0.21:3.}
def _add_data_vars(sd, trls, var_names, check=True,
                   rwd_conversion=rwd_conversion):
    if check:
        all_trls = np.stack(list(t[0, 0][0] for t in trls[:, 0]),
                            axis=0)
        all_trls[np.isnan(all_trls)] = -1000
        if not np.all(all_trls == all_trls):
            mask = np.logical_not(all_trls[0:1] == all_trls)
            ind = np.where(mask)
            print(var_names[ind[-1]])
            print(all_trls.shape)
            print(all_trls[mask].shape)
            print(all_trls[mask])
        # assert np.all(all_trls[0:1] == all_trls)
    for i, vn in enumerate(var_names):
        vn_vals = trls[0, 0][0, 0][0][:, i]
        if vn[0].split(' ')[0] == 'rwd' and vn_vals[0] < 1:
            vn_vals = np.array(list(rwd_conversion[vn_i] for vn_i in vn_vals))
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

subj_split_fields = (('probability weighted offer 1 JMF',
                      'probability weighted offer 2 JMF'),
                     ('reward weighted offer 1 JMF',
                      'reward weighted offer 2 JMF'),
                     ('subjective utlity offer 1 JMF',
                      'subjective utlity offer 2 JMF'))
subj_new_fields = ('subj_prob_{}', 'subj_rwd_{}', 'subj_ev_{}')

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

def _infer_sessions(trl_nums):
    masks = []
    tnum = trl_nums[0]
    curr_mask = [0]
    for i in range(1, len(trl_nums)):
        tnum = trl_nums[i]
        if tnum == trl_nums[curr_mask[-1]]:
            curr_mask.append(i)
        else:
            masks.append(curr_mask)
            curr_mask = [i]
    masks.append(curr_mask)
    binary_masks = np.zeros((len(masks), len(trl_nums)), dtype=bool)
    for i, mask in enumerate(masks):
        binary_masks[i, mask] = True
    return masks

def load_fine_data(folder, regions_list=('OFC', 'PCC', 'pgACC', 'vmPFC', 'VS'),
                   tv_templ='TrialInfo/{}_Trial_Vars.mat',
                   psth_templ='Neural/{}_psth_rebinned_20ms_original.mat',
                   tv_key='Trial_Vars', psth_key='data',
                   sub_key='subj_ID', vn_key='Variable_Names',
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
            trl_nums = np.array(list(pij[0][0, 0][0].shape[0]
                                     for pij in trls_ij))
            psth_trl_nums = np.array(list(pij[0]['psth'][0, 0].shape[0]
                                     for pij in psth_ij))
            session_masks = _infer_sessions(psth_trl_nums)
            # session_masks = np.identity(len(psth_trl_nums)).astype(bool)
            assert np.all(trl_nums == psth_trl_nums)
            for k, mask in enumerate(session_masks):
                tn_u = trl_nums[mask][0]
                session_dict = {} 
                try:
                
                    pop = np.stack(list(pij[0]['psth'][0, 0]
                                        for pij in psth_ij[mask]),
                                   axis=1)
                    session_dict['neur_regions'] = ((region,)*pop.shape[1],)*pop.shape[0]
                    session_dict['psth'] = list(pop)
                    session_dict = _add_data_vars(session_dict, trls_ij[mask],
                                                  var_names)
                    timing_dict = {ename[0]:np.ones(len(pop))*etime[0, 0]
                                   for (ename, etime) in event_times}
                    session_dict.update(timing_dict)
                    session_dict = _make_convenience_data_vars(session_dict)
                    session_dict = _make_convenience_data_vars(
                        session_dict, split_fields=subj_split_fields,
                        new_fields=subj_new_fields)
                    session_dict = _make_convenience_data_vars(
                        session_dict, offer1_side=chosen_offer_field,
                        split_keys=('chosen', 'unchosen'))
                    session_dict = _make_convenience_data_vars(
                        session_dict, split_fields=subj_split_fields,
                        new_fields=subj_new_fields,
                        offer1_side=chosen_offer_field,
                        split_keys=('chosen', 'unchosen'))
 
                    k1 = 'Expected value offer 1'
                    session_dict['ev offer 1'] = session_dict[k1]
                    k2 = 'Expected value offer 2'
                    session_dict['ev offer 2'] = session_dict[k2]
                    session_frame = pd.DataFrame.from_dict(session_dict)

                    trl_num = np.arange(1, len(session_dict[k2]) + 1)
                    session_dict["trial_num"] = trl_num

                    k1 = 'subjective utlity offer 1 JMF'
                    session_dict['subj_ev offer 1'] = session_dict[k1]
                    k2 = 'subjective utlity offer 2 JMF'
                    session_dict['subj_ev offer 2'] = session_dict[k2]
                    session_frame = pd.DataFrame.from_dict(session_dict)

                    n_neurs.append(pop.shape[1])
                    datas.append(session_frame)
                    regions.append(region)
                    animals.append(an)
                    psth_timings.append(psth_time_vec)
                    dates.append('{}-{}-{}'.format(region, an, tn_u))
                except AssertionError:
                    print(region, an, trl_nums[mask], 'failed')
    super_dict = dict(date=dates, animal=animals, data=datas, n_neurs=n_neurs,
                      psth_timing=psth_timings)
    return super_dict
