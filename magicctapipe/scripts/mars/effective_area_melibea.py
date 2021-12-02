import numpy as np

from astropy.table import QTable, vstack
import astropy.units as u

import uproot
from pyirf.simulations import SimulatedEventsInfo

melibea_columns = {
        'event_id': ('MMcEvt_1.fEvtNumber', dict(dtype=int)),
        'true_energy': ('MMcEvt_1.fEnergy', dict(unit=u.GeV)),
        'reco_energy': ('MEnergyEst.fEnergy', dict(unit=u.GeV)),
        'hadroness': ('MHadronness.fHadronness', dict()),
        'size1': ('MHillas_1.fSize', dict()),
        'size2': ('MHillas_2.fSize', dict()),
        'reco_source_x': ('MStereoParDisp.fDirectionX', dict(unit=u.deg)),
        'reco_source_y': ('MStereoParDisp.fDirectionY', dict(unit=u.deg)),
        'pointing_zen': ('MMcEvt_1.fTelescopeTheta', dict(unit=u.rad)),
        'pointing_az': ('MMcEvt_1.fTelescopePhi', dict(unit=u.rad)),
        'true_zen': ('MMcEvt_1.fTheta', dict(unit=u.rad)),
        'true_az': ('MMcEvt_1.fPhi', dict(unit=u.rad)),
        'particle_id': ('MMcEvt_1.fPartId', dict()),
        'theta2' : ('MStereoParDisp.fTheta2', dict(unit=u.deg**2))
}

def read_dl2_files(paths):
    events = []
    sim_info = None

    for path in paths:
        e, s = read_dl2_file(path)
        events.append(e)

        if sim_info is None:
            sim_info = s
        else:
            for attr in ('energy_min', 'energy_max', 'max_impact', 'viewcone', 'spectral_index'):
                assert getattr(s, attr) == getattr(sim_info, attr),  f'{attr} does not match'

            sim_info.n_showers += s.n_showers

    events = vstack(events)

    return events, sim_info

def read_dl2_file(path):
    f = uproot.open(path)

    events_tree = f['Events']
    events = QTable()

    for column, (branch, kwargs) in melibea_columns.items():
        events[column] = u.Quantity(events_tree[branch].array(library="np"), copy=False, **kwargs)

    events['reco_source_fov_offset'] = np.sqrt(events['theta2'])
    events['theta'] = events['reco_source_fov_offset']

    # mars weight is rate per second, we want expected event count for observation time
    # events['mars_weight'] *= T_OBS.to_value(u.s)

    for col in ['true', 'pointing']:
        events[col + '_alt'] = (np.pi / 2) * u.rad - events[col + '_zen']

    run_header_tree = f['RunHeaders']

    spectral_index = run_header_tree['MMcCorsikaRunHeader.fSlopeSpec'].array(library="np")
    view_cone = run_header_tree['MMcCorsikaRunHeader.fViewconeAngles[2]'].array(library="np")
    e_low = run_header_tree['MMcCorsikaRunHeader.fELowLim'].array(library="np")
    e_high = run_header_tree['MMcCorsikaRunHeader.fEUppLim'].array(library="np")
    max_impact = run_header_tree['MMcRunHeader_1.fImpactMax'].array(library="np")

    n_showers = np.sum(run_header_tree['MMcRunHeader_1.fNumSimulatedShowers'].array(library="np"))

    # check all simulations are compatible
    for array in (spectral_index, e_low, e_high, max_impact, view_cone[:, 0], view_cone[:, 1]):
        assert len(np.unique(array[:])) == 1

    sim_info = SimulatedEventsInfo(
        n_showers=n_showers,
        spectral_index=spectral_index[0],
        energy_min=e_low[0] * u.GeV,
        energy_max=e_high[0] * u.GeV,
        max_impact=(max_impact[0] * u.cm).to(u.m),
        viewcone=view_cone[0, 1] * u.deg,
    )

    return events, sim_info

def find_hadronness_cuts(events, minsize, minzen, maxzen, mineest, maxeest, nbinsE, nbinsH, minH = 0.15, maxH = 0.95, hadEff = 0.9):
    print("=========== HADRONNESS CUTS CALCULATION ===========")
    print(f"Number of events before events selection: {len(events.index)}")

    event_cuts_h = f"(size1 > {minsize}) & (size2 > {minsize}) & (pointing_zen < {maxzen}) & (pointing_zen > {minzen}) & (reco_energy < {maxeest}) & (reco_energy > {mineest})"
    print(f"Event cuts: {event_cuts_h}")

    reco_valid = events[events["reco_energy"] > 0]

    reco_selected_h = reco_valid.query(event_cuts_h)
    print(f"Number of events after events selection: {len(reco_selected_h.index)}")

    e_step = (np.log10(maxeest) - np.log10(mineest))/nbinsE
    h_step = (1.01 + 0.01)/nbinsH
    had_vs_e, had_vs_e_xe, had_vs_e_ye = np.histogram2d(np.log10(reco_selected_h["reco_energy"]), reco_selected_h["hadroness"], bins=[nbinsE+1, nbinsH+1], range=[[np.log10(mineest)-e_step, np.log10(maxeest)], [-0.01-h_step, 1.01]])

    had_vs_e = had_vs_e.T

    e_vs_had = had_vs_e.T

    hadcuts = np.repeat(-1.0, nbinsE)
    print("Hadronness cuts:")
    for ebin in range(1,nbinsE+1):
        project = e_vs_had[ebin]
        #print(f"Slice integral: {np.sum(project)}")

        if np.sum(project) < 1:
            print(f"Energy bin {ebin} ({10**had_vs_e_xe[ebin]:.3f} < E < {10**had_vs_e_xe[ebin+1]:.3f} GeV): {hadcuts[ebin-1]:.5f}")
            continue

        if np.sum(project) < 10:
            if ebin > 1 and hadcuts[ebin-2]>0:
                hadcuts[ebin-1] = hadcuts[ebin-2]
            else:
                hadcuts[ebin-1] = maxH
            print(f"Energy bin {ebin} ({10**had_vs_e_xe[ebin]:.3f} < E < {10**had_vs_e_xe[ebin+1]:.3f} GeV): {hadcuts[ebin-1]:.5f}")
            continue

        for hbin in range(1,nbinsH+1):
            if np.sum(project[1:hbin+1]) > 0.9*np.sum(project):
                hadcuts[ebin-1] = min(max(had_vs_e_ye[hbin+1],minH), maxH)
                break

        print(f"Energy bin {ebin} ({10**had_vs_e_xe[ebin]:.3f} < E < {10**had_vs_e_xe[ebin+1]:.3f} GeV): {hadcuts[ebin-1]:.5f}")

    return hadcuts, had_vs_e_xe

def find_theta2_cuts(events, minsize, minzen, maxzen, mineest, maxeest, nbinsE, nbinsTH, hadcuts, had_vs_e, minTH = 0.01, maxTH = 0.2, thEff = 0.75):
    print("=========== THETA2 CUTS CALCULATION ===========")
    print(f"Number of events before events selection: {len(events.index)}")

    event_cuts_th = f"(size1 > {minsize}) & (size2 > {minsize}) & (reco_energy < {maxeest}) & (reco_energy > {mineest})"
    print(f"Event cuts: {event_cuts_th}")

    reco_valid = events[events["reco_energy"] > 0]
    reco_selected_th = reco_valid.query(event_cuts_th)

    hadcut = []
    for logenergy in np.log10(reco_selected_th["reco_energy"].values):
        ebin = np.where(had_vs_e>logenergy)[0][0] - 1
        if ebin == 0:
            ebin = 1
        if ebin == nbinsE+1:
            ebin = nbinsE
        hadcut.append(hadcuts[ebin-1])
    reco_selected_th["hadcut"] = hadcut

    reco_selected_th = reco_selected_th[reco_selected_th["hadroness"] <= reco_selected_th["hadcut"]]
    print(f"Number of events after events selection: {len(reco_selected_th.index)}")

    e_step = (np.log10(maxeest) - np.log10(mineest))/nbinsE
    th_step = (0.4)/nbinsTH
    th_vs_e, th_vs_e_xe, th_vs_e_ye = np.histogram2d(np.log10(reco_selected_th["reco_energy"]), reco_selected_th["theta2"], bins=[nbinsE+1, nbinsTH+1], range=[[np.log10(mineest)-e_step, np.log10(maxeest)], [0-th_step, 0.4]])

    th_vs_e = th_vs_e.T
    e_vs_th = th_vs_e.T

    thetacuts = np.repeat(-1.0, nbinsE)
    print("Theta2 cuts:")
    for ebin in range(1,nbinsE+1):
        project = e_vs_th[ebin]

        if np.sum(project) < 1:
            print(f"Energy bin {ebin} ({10**th_vs_e_xe[ebin]:.3f} < E < {10**th_vs_e_xe[ebin+1]:.3f} GeV): {thetacuts[ebin-1]:.3f}")
            continue

        if np.sum(project) < 10:
            if ebin > 1 and thetacuts[ebin-2]>0:
                thetacuts[ebin-1] = thetacuts[ebin-2]
            else:
                thetacuts[ebin-1] = maxTH
            print(f"Energy bin {ebin} ({10**th_vs_e_xe[ebin]:.3f} < E < {10**th_vs_e_xe[ebin+1]:.3f} GeV): {thetacuts[ebin-1]:.3f}")
            continue

        for thbin in range(1,nbinsTH+1):
            if np.sum(project[1:thbin+1]) > thEff*np.sum(project):
                thetacuts[ebin-1] = min(max(th_vs_e_ye[thbin+1],minTH), maxTH)
                break

        print(f"Energy bin {ebin} ({10**th_vs_e_xe[ebin]:.3f} < E < {10**th_vs_e_xe[ebin+1]:.3f} GeV): {thetacuts[ebin-1]:.3f}")

    return thetacuts

p = dict()
p['events'], p['simulation_info'] = read_dl2_files(["/storage/gpfs_data/ctalocal/aberti/Analysis_MAGIC/OFFS/Coach_ST0312/05to70/Melibea_MC/GA_za05to35_8_Q_w0_2.root"])
gammas = p['events']

reco = gammas.to_pandas()

minsize = 50.0
minzen  = u.Quantity(5.0*u.deg, u.rad).value
maxzen  = u.Quantity(20.0*u.deg, u.rad).value
mineest = u.Quantity(4.64159*u.GeV, u.GeV).value
maxeest = u.Quantity(46415.9*u.GeV, u.GeV).value
nbinsE = 30
nbinsH = 4080
nbinsTH = 40

hadcuts, had_vs_e = find_hadronness_cuts(reco, minsize, minzen, maxzen, mineest, maxeest, nbinsE, nbinsH)
theta2cuts        = find_theta2_cuts(reco, minsize, minzen, maxzen, mineest, maxeest, nbinsE, nbinsTH, hadcuts, had_vs_e)
