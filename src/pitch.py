import numpy as np

def get_prop_octave(pitch, o=1):
    madyhama = len(np.where(np.logical_and(pitch>=0, pitch<=1200))[0]) # middle octave
    tara = len(np.where(pitch>1200)[0]) # higher octave
    mandhira = len(np.where(pitch<0)[0]) # lower octave

    octs = [mandhira, madyhama, tara]
    return octs[o]/len(pitch)

def transpose_pitch(pitch):
    ## WARNING: Assumes no pitch values in middle octave+2 or middle octave-2
    ## and that no svara traverses two octaves (likely due to pitch errors)
    r_prop = get_prop_octave(pitch, 0)
    p_prop = get_prop_octave(pitch, 1)
    i_prop = get_prop_octave(pitch, 2)

    if r_prop == 0 and i_prop == 0:
        # no transposition
        return pitch, False

    if r_prop == 0 and p_prop == 0:
        # transpose down
        return pitch-1200, True
    
    if i_prop == 0 and p_prop == 0:
        # transpose up
        return pitch+1200, True

    if i_prop > 0.8:
        # transpose down
        return pitch-1200, True

    return pitch, False