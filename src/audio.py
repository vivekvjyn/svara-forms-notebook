import soundfile as sf

def save_audio(annotations, vocal, annot_ix, audio_path, sr=44100):

    curr_row = annotations.iloc[annot_ix]

    s1 = round(curr_row['start_sec']*sr)
    s2 = round(curr_row['end_sec']*sr)
    
    vsamp = vocal[s1:s2]

    sf.write(audio_path, vsamp, sr)