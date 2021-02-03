import math
import numpy as np
from astropy import units as u

from ctapipe.containers import ReconstructedShowerContainer

def camera_to_direction(rc, CTphi, CTtheta, x, y):
    #
    # convention of phi defined in TDAS 01-05
    #
    sinphi   = math.sin(CTphi)
    cosphi   = math.cos(CTphi)
    costheta = math.cos(CTtheta)
    sintheta = math.sin(CTtheta)

    xc = x/rc
    yc = y/rc

    norm = 1/math.sqrt(1+xc*xc+yc*yc)

    xref = xc * norm
    yref = yc * norm
    zref = -1 * norm

    cosx =  xref * sinphi + yref * costheta*cosphi - zref * sintheta*cosphi
    cosy = -xref * cosphi + yref * costheta*sinphi - zref * sintheta*sinphi
    cosz =                  yref * sintheta        + zref * costheta

    cosy *= -1
    cosz *= -1

    return cosx, cosy, cosy

def stereo_par_calc_mars(hillas_params_dict, subarray, telescope_pointing_dict):
    is_valid = False

    rad_to_deg = 180./np.pi

    #
    # Pointing position of the telescopes
    # We should have a option to use or not the starguider correction
    #
    M1_pointing_Az = telescope_pointing_dict[1].az.value
    M1_pointing_Zd = np.pi - telescope_pointing_dict[1].alt.value
    M2_pointing_Az = telescope_pointing_dict[2].az.value
    M2_pointing_Zd = np.pi - telescope_pointing_dict[2].alt.value

    # Start Stereo-reconstruction
    #
    # First
    # Get the direction corresponding to the c.o.g. of the image on the camera

    M1_cosx_a = math.nan
    M1_cosy_a = math.nan
    M1_cosz_a = math.nan # Direction from M1 to the shower c.o.g.

    camera_dist = 17.0*1000

    cog_x = hillas_params_dict[1].x.value.to("mm")
    cog_y = hillas_params_dict[1].y.value.to("mm")

    M1_cosx_a, M1_cosy_a, M1_cosz_a = camera_to_direction(camera_dist, M1_pointing_Az,M1_pointing_Zd, cog_x, cog_y)

    # Now we get another (arbitrary) point along the image long axis,
    # fMeanX + cosdelta, fMeanY + sindelta, and calculate the direction
    # to which it corresponds.

    cosx_b = math.nan
    cosy_b = math.nan
    cosz_b = math.nan

    psi = hillas_params_dict[1].psi.value.to("rad")
    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)

    cosx_b, cosy_b, cosz_b = camera_to_direction(camera_dist, M1_pointing_Az, M1_pointing_Zd, cog_x+cos_psi, cog_y+sin_psi)

    #
    # The vectorial product of the latter two vectors is a vector
    # perpendicular to the plane which contains the shower axis and
    # passes through the telescope center (center of reflector).

    M1_SPlanVectX = M1_cosy_a*cosz_b - cosy_b*M1_cosz_a
    M1_SPlanVectY = M1_cosz_a*cosx_b - cosz_b*M1_cosx_a
    M1_SPlanVectZ = M1_cosx_a*cosy_b - cosx_b*M1_cosy_a

    # Now, same thing for the second telescope

    M2_cosx_a = math.nan
    M2_cosy_a = math.nan
    M2_cosz_a = math.nan # Direction from M2 to the shower c.o.g.

    cog_x = hillas_params_dict[2].x.value.to("mm")
    cog_y = hillas_params_dict[2].y.value.to("mm")

    M2_cosx_a, M2_cosy_a, M2_cosz_a = camera_to_direction(camera_dist, M2_pointing_Az, M2_pointing_Zd, cog_x, cog_y)

    cosx_b = math.nan
    cosy_b = math.nan
    cosz_b = math.nan

    psi = hillas_params_dict[2].psi.value.to("rad")
    cos_psi = math.cos(psi)
    sin_psi = math.sin(psi)

    cosx_b, cosy_b, cosz_b = camera_to_direction(camera_dist, M2_pointing_Az, M2_pointing_Zd, cog_x+cos_psi, cog_y+sin_psi)

    M2_SPlanVectX = M2_cosy_a*cosz_b - cosy_b*M2_cosz_a
    M2_SPlanVectY = M2_cosz_a*cosx_b - cosz_b*M2_cosx_a
    M2_SPlanVectZ = M2_cosx_a*cosy_b - cosx_b*M2_cosy_a

    # The vectorial product of the 2 Vectors M1_SPlanVect and M2_SPlanVec
    # is a vector contained in both planes. It has the same direction as the shower.

    showerdir_cosx = M1_SPlanVectY*M2_SPlanVectZ - M2_SPlanVectY*M1_SPlanVectZ
    showerdir_cosy = M1_SPlanVectZ*M2_SPlanVectX - M2_SPlanVectZ*M1_SPlanVectX
    showerdir_cosz = M1_SPlanVectX*M2_SPlanVectY - M2_SPlanVectX*M1_SPlanVectY
    norm = math.sqrt(showerdir_cosx*showerdir_cosx + showerdir_cosy*showerdir_cosy + showerdir_cosz*showerdir_cosz)

    if (norm==0):
        return ReconstructedShowerContainer()
    if (showerdir_cosz<0):
        norm*=-1. # to have the vector pointing up

    showerdir_cosx*=1./norm
    showerdir_cosy*=1./norm
    showerdir_cosz*=1./norm

    fDirectionZd = math.acos(showerdir_cosz)*rad_to_deg
    fDirectionAz = math.atan2(-showerdir_cosy,showerdir_cosx)*rad_to_deg

    #
    # Estimate core position:
    #
    # The vectorial product of M1_SPlanVect and (0,0,1) is a vector on
    # the horizontal plane pointing from the telescope center to the
    # shower core position on the z= M1_z plane (telescope level).
    #
    # or from the projection of M1 to z=0, to the core position at z=0 (ground level)
    #
    #    Float_t M1_coreVersorX =  M1_SPlanVectY
    #    Float_t M1_coreVersorY = -M1_SPlanVectX
    #
    #    Float_t M2_coreVersorX =   M2_SPlanVectY
    #    Float_t M2_coreVersorY =  -M2_SPlanVectX
    #
     
    #get telescope position:
    M1_x = subarray.positions[1][0].value.to("cm")
    M1_y = subarray.positions[1][1].value.to("cm")
    M1_z = subarray.positions[1][2].value.to("cm")
    M2_x = subarray.positions[2][0].value.to("cm")
    M2_y = subarray.positions[2][1].value.to("cm")
    M2_z = subarray.positions[2][2].value.to("cm")

    # Correction of M1 and M2 positions in the z=0 plan if M1_z or M2_z is not null
    # if(fabs(M1_z)>1.){ 
    M1proj_x = M1_x - M1_z*showerdir_cosx/showerdir_cosz
    M1proj_y = M1_y - M1_z*showerdir_cosy/showerdir_cosz
    # }
    # if(fabs(M2_z)>1.){ 
    M2proj_x = M2_x - M2_z*showerdir_cosx/showerdir_cosz
    M2proj_y = M2_y - M2_z*showerdir_cosy/showerdir_cosz
    # }

    fCoreX = (M2proj_y - M1proj_y)*M2_SPlanVectY*M1_SPlanVectY + M2proj_x*M2_SPlanVectX*M1_SPlanVectY - M1proj_x*M1_SPlanVectX*M2_SPlanVectY
    fCoreY = (M2proj_x - M1proj_x)*M1_SPlanVectX*M2_SPlanVectX + M2proj_y*M2_SPlanVectY*M1_SPlanVectX - M1proj_y*M1_SPlanVectY*M2_SPlanVectX
    den = M2_SPlanVectY*M1_SPlanVectX - M1_SPlanVectY*M2_SPlanVectX
    if(den==0):
        return ReconstructedShowerContainer()
    fCoreX /= -den
    fCoreY /= den

    #
    # Calculate impact parameters
    #
    # The norm of the Projection of M1-core vector on the shower axis is the scalar product M1_core*Showerdir
    scalar = (fCoreX-M1_x)*showerdir_cosx + (fCoreY-M1_y)*showerdir_cosy - M1_z*showerdir_cosz
    fM1Impact = (fCoreX-M1_x)*(fCoreX-M1_x) + (fCoreY-M1_y)*(fCoreY-M1_y) + M1_z*M1_z - scalar*scalar
    if(fM1Impact<0.):
        return ReconstructedShowerContainer()
    fM1Impact = math.sqrt(fM1Impact)

    scalar = (fCoreX-M2_x)*showerdir_cosx + (fCoreY-M2_y)*showerdir_cosy - M2_z*showerdir_cosz
    fM2Impact = (fCoreX-M2_x)*(fCoreX-M2_x) + (fCoreY-M2_y)*(fCoreY-M2_y) + M2_z*M2_z - scalar*scalar
    if(fM2Impact<0.):
        return ReconstructedShowerContainer()
    fM2Impact = math.sqrt(fM2Impact)

    # Same technique for the E-W and N-S components:
    # And calculate the azimuth of the projected impact
    fM1Impact_EW = (fCoreY-M1_y)* math.sqrt(1-showerdir_cosy*showerdir_cosy)
    fM1Impact_NS = (fCoreX-M1_x)* math.sqrt(1-showerdir_cosx*showerdir_cosx)
    fM1ImpactAz  = math.atan2(-fM1Impact_EW,fM1Impact_NS)*rad_to_deg

    fM2Impact_EW = (fCoreY-M2_y)* math.sqrt(1-showerdir_cosy*showerdir_cosy)
    fM2Impact_NS = (fCoreX-M2_x)* math.sqrt(1-showerdir_cosx*showerdir_cosx)
    fM2ImpactAz  = math.atan2(-fM2Impact_EW,fM2Impact_NS)*rad_to_deg

    #
    # Compute Shower Height Maximum
    #
    fMaxHeight = -1

    #
    # We know 3 straights which should contain the shower Maximum
    # M1-ShowerMax
    # M2-ShowerMax
    # Impact-ShowerMax
    #
    # As life is in 3D, they may not intersect.
    # Lets assume that the height of the shower max is the height of the plane
    # in which the triangle made by the 3 points has the smallest circonference
    #

    #For Telescope 1:  Equations of M1-ShowerMax:
    # X = a1*(Z-M1_z) + M1_x = a1 Z + (M1_x-a1*M1_z)
    # Y = c1*(Z-M1_z) + M1_y = c1 Z + (M1_y-c1*M1_z)
    if(M1_cosz_a==0):
        return ReconstructedShowerContainer()
    a1 = M1_cosx_a/M1_cosz_a
    c1 = M1_cosy_a/M1_cosz_a

    #For Telescope 2:  Equations of M2-ShowerMax:
    # X = a2*(Z-M2_z) + M2_x = a2 Z + (M2_x-a2*M2_z)
    # Y = c2*(Z-M2_z) + M2_y = c2 Z + (M2_y-c2*M2_z)
    if(M2_cosz_a==0):
        return ReconstructedShowerContainer()
    a2 = M2_cosx_a/M2_cosz_a
    c2 = M2_cosy_a/M2_cosz_a

    #From the core impact:  Equations of Impact-ShowerMax:
    # X = a3 Z + fCoreX
    # Y = c3 Z + fCoreY
    if(showerdir_cosz==0):
        return ReconstructedShowerContainer()
    a3 = showerdir_cosx/showerdir_cosz
    c3 = showerdir_cosy/showerdir_cosz

    # to use previous calculation ignoring telescope Z, we indroduce the values:
    M1proj_x = M1_x-a1*M1_z
    M1proj_y = M1_y-c1*M1_z
    M2proj_x = M2_x-a2*M2_z
    M2proj_y = M2_y-c2*M2_z

    #
    # After resolution a simple equation , we find that the Shower Max hight should be:
    #

    fMaxHeight = -((a1-a2)*(M1proj_x-M2proj_x) + (c1-c2)*(M1proj_y-M2proj_y) + (a2-a3)*(M2proj_x-fCoreX)+ (c2-c3)*(M2proj_y-fCoreY) + (a3-a1)*(fCoreX-M1proj_x)+ (c3-c1)*(fCoreY-M1proj_y))
    fMaxHeight /= ((a1-a2)*(a1-a2) + (c1-c2)*(c1-c2) + (a2-a3)*(a2-a3) + (c2-c3)*(c2-c3) + (a3-a1)*(a3-a1) + (c3-c1)*(c3-c1))

    is_valid = True
    result = ReconstructedShowerContainer(
        alt=(90. - fDirectionZd)*u.deg,
        az=-fDirectionAz*u.deg,
        core_x=(fCoreX/100.0)*u.m,
        core_y=(fCoreY/100.0)*u.m,
        tel_ids=[h for h in hillas_params_dict.keys()],
        average_intensity=np.mean([h.intensity for h in hillas_params_dict.values()]),
        is_valid=is_valid,
        h_max=(fMaxHeight/100.0)*u.m,
    )

    return result