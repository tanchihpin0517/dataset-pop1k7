import os
import glob
import pickle
import numpy as np
import miditoolkit
from miditoolkit.midi.containers import Note, TempoChange
import collections
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp

# ================================================== #  
#  Configuration                                     #
# ================================================== #  
BEAT_RESOL = 480
BAR_RESOL = BEAT_RESOL * 4
SUBBEAT_RESOL = BEAT_RESOL // 4
TICK_RESOL = BEAT_RESOL // 4
# TICK_RESOL = BEAT_RESOL // 32

INSTR_NAME_MAP = {'piano': 0}
MIN_BPM = 40
MIN_VELOCITY = 40
NOTE_SORTING = 1 #  0: ascending / 1: descending

DEFAULT_VELOCITY_BINS = np.linspace(0,  128, 64+1, dtype=int)
DEFAULT_BPM_BINS      = np.linspace(32, 224, 64+1, dtype=int)
DEFAULT_SHIFT_BINS    = np.linspace(-60, 60, 60+1, dtype=int)
DEFAULT_DURATION_BINS = np.arange(
        BEAT_RESOL/8, BEAT_RESOL*8+1, BEAT_RESOL/8)

# ================================================== #  


def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list


def proc_one_map(args):
    path_midi, path_outfile = args

    global TICK_RESOL

    TICK_RESOL = SUBBEAT_RESOL
    proc_remi(path_midi, path_outfile)
    TICK_RESOL = SUBBEAT_RESOL // 2
    proc_remi(path_midi, path_outfile)
    TICK_RESOL = SUBBEAT_RESOL // 3
    proc_remi(path_midi, path_outfile)
    TICK_RESOL = SUBBEAT_RESOL // 12
    # TICK_RESOL = BEAT_RESOL // 12
    proc_remi(path_midi, path_outfile)

    # TICK_RESOL = SUBBEAT_RESOL // 60
    # proc_remi(path_midi, path_outfile)

def proc_remi(path_midi, path_outfile):
    # --- load --- #
    midi_obj = miditoolkit.midi.parser.MidiFile(path_midi)

    # load notes
    instr_notes = collections.defaultdict(list)
    for instr in midi_obj.instruments:
        # skip 
        if instr.name not in INSTR_NAME_MAP.keys():
            continue

        # process
        instr_idx = INSTR_NAME_MAP[instr.name]
        for note in instr.notes:
            note.instr_idx=instr_idx
            instr_notes[instr_idx].append(note)
        if NOTE_SORTING == 0:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, x.pitch))
        elif NOTE_SORTING == 1:
            instr_notes[instr_idx].sort(
                key=lambda x: (x.start, -x.pitch))
        else:
            raise ValueError(' [x] Unknown type of sorting.')

    # load chords
    chords = []
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] != 'global' and \
        'Boundary' not in marker.text.split('_')[0]:
            chords.append(marker)
    chords.sort(key=lambda x: x.time)

    # load tempos
    tempos = midi_obj.tempo_changes
    tempos.sort(key=lambda x: x.time)

    # load labels
    labels = []
    for marker in midi_obj.markers:
        if 'Boundary' in marker.text.split('_')[0]:
            labels.append(marker)
    labels.sort(key=lambda x: x.time)

    # load global bpm
    gobal_bpm = 120
    for marker in midi_obj.markers:
        if marker.text.split('_')[0] == 'global' and \
            marker.text.split('_')[1] == 'bpm':
            gobal_bpm = int(marker.text.split('_')[2])

    # --- process items to grid --- #
    # compute empty bar offset at head
    first_note_time = min([instr_notes[k][0].start for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1].start for k in instr_notes.keys()])

    quant_time_first = int(np.round(first_note_time  / TICK_RESOL) * TICK_RESOL)
    offset = quant_time_first // BAR_RESOL # empty bar
    last_bar = int(np.ceil(last_note_time / BAR_RESOL)) - offset
    # print(' > offset:', offset)
    # print(' > last_bar:', last_bar)

    # process notes
    intsr_gird = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            note.start = note.start - offset * BAR_RESOL
            note.end = note.end - offset * BAR_RESOL

            # quantize start
            quant_time = int(np.round(note.start / TICK_RESOL) * TICK_RESOL)

            # velocity
            note.velocity = DEFAULT_VELOCITY_BINS[
                np.argmin(abs(DEFAULT_VELOCITY_BINS-note.velocity))]
            note.velocity = max(MIN_VELOCITY, note.velocity)

            # shift of start
            note.shift = note.start - quant_time 
            note.shift = DEFAULT_SHIFT_BINS[np.argmin(abs(DEFAULT_SHIFT_BINS-note.shift))]

            # duration
            note_duration = note.end - note.start
            if note_duration > BAR_RESOL:
                note_duration = BAR_RESOL
            ntick_duration = int(np.round(note_duration / TICK_RESOL) * TICK_RESOL)
            ntick_duration = max(ntick_duration, TICK_RESOL)
            note.duration = ntick_duration

            # append
            note_grid[quant_time].append(note)

        # set to track
        intsr_gird[key] = note_grid.copy()

    # process chords
    chord_grid = collections.defaultdict(list)
    for chord in chords:
        # quantize
        chord.time = chord.time - offset * BAR_RESOL
        chord.time  = 0 if chord.time < 0 else chord.time 
        quant_time = int(np.round(chord.time / TICK_RESOL) * TICK_RESOL)

        # append
        chord_grid[quant_time].append(chord)

    # process tempo
    tempo_grid = collections.defaultdict(list)
    for tempo in tempos:
        # quantize
        tempo.time = tempo.time - offset * BAR_RESOL
        tempo.time = 0 if tempo.time < 0 else tempo.time
        quant_time = int(np.round(tempo.time / TICK_RESOL) * TICK_RESOL)
        tempo.tempo = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-tempo.tempo))]

        # append
        tempo_grid[quant_time].append(tempo)

    # process boundary
    label_grid = collections.defaultdict(list)
    for label in labels:
        # quantize
        label.time = label.time - offset * BAR_RESOL
        label.time = 0 if label.time < 0 else label.time
        quant_time = int(np.round(label.time / TICK_RESOL) * TICK_RESOL)

        # append
        label_grid[quant_time] = [label]
        
    # process global bpm
    gobal_bpm = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS-gobal_bpm))]

    # collect
    song_data = {
        'notes': intsr_gird,
        'chords': chord_grid,
        'tempos': tempo_grid,
        'labels': label_grid,
        'metadata': {
            'global_bpm': gobal_bpm,
            'last_bar': last_bar,
        }
    }


    # save
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)
    # pickle.dump(song_data, open(path_outfile+".pkl", 'wb'))

    if not Path(path_outfile+f"_{TICK_RESOL}.mid").exists():
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = BEAT_RESOL
        track = miditoolkit.midi.containers.Instrument(program=0, is_drum=False, name='piano')
        midi.instruments = [track]

        for key, inst in song_data['notes'].items():
            for tick, notes in inst.items():
                for note in notes:
                    for note in notes:
                        # create one note
                        start = tick
                        end = tick + note.duration
                        pitch = note.pitch
                        velocity = note.velocity
                        n = miditoolkit.Note(start=start, end=end, pitch=pitch, velocity=velocity)
                        midi.instruments[0].notes.append(n)

        for tick, tempos in song_data['tempos'].items():
            for tempo in tempos:
                m = miditoolkit.midi.containers.TempoChange(time=tick, tempo=tempo.tempo)
                midi.tempo_changes.append(m)
        midi.dump(path_outfile+f"_{TICK_RESOL}.mid")

    remi = song_to_remi(song_data)
    Path(path_outfile+f"_{TICK_RESOL}.txt").write_text('\n'.join(remi))
    midi = remi_to_midi(remi)
    midi.dump(path_outfile+f"_{TICK_RESOL}_remi.mid")

def song_to_remi(song):
    events = []

    for tick, tempos in song['tempos'].items():
        for tempo in tempos:
            events.append((tick, tempo))
            assert tick % SUBBEAT_RESOL == 0

    for inst, notes in song['notes'].items():
        for tick, notes in song['notes'][inst].items():
            for note in notes:
                events.append((tick, note))

    events.sort(key=lambda x: x[0]) # `sort` is stable
    remi = ['bar']
    bar_tick = 0
    last_subbeat = -1
    last_shift = -1

    for tick, e in events:
        if tick - bar_tick >= BAR_RESOL:
            bar_tick = (tick // BAR_RESOL) * BAR_RESOL
            remi.append('bar')

        subbeat = ((tick - bar_tick) // SUBBEAT_RESOL) * SUBBEAT_RESOL
        if subbeat != last_subbeat:
            remi.append(f'beat_{subbeat}')
            last_subbeat = subbeat
            last_shift = -1

        shift = round((tick % SUBBEAT_RESOL) / TICK_RESOL) * TICK_RESOL
        # shift = tick % SUBBEAT_RESOL
        if shift != last_shift:
            if shift != 0:
                remi.append(f'shift_{shift}')
            last_shift = shift
        # if shift != 0:
        #     remi.append(f'shift_{shift}')

        if isinstance(e, TempoChange):
            remi.append(f'tempo_{e.tempo}')
        if isinstance(e, Note):
            dur = max(round(e.duration / SUBBEAT_RESOL), 1) * SUBBEAT_RESOL
            remi.append(f'pitch_{e.pitch}')
            remi.append(f'velocity_{e.velocity}')
            remi.append(f'duration_{dur}')

    return remi

def remi_to_midi(remi):
    midi = miditoolkit.midi.parser.MidiFile()
    midi.ticks_per_beat = BEAT_RESOL
    track = miditoolkit.midi.containers.Instrument(program=0, is_drum=False, name='piano')
    midi.instruments = [track]

    bar_tick = 0
    beat_tick = 0
    shift = 0

    pitch, velocity, duration = 0, 0, 0

    for event in remi:
        if event == 'bar':
            bar_tick += BAR_RESOL
            beat_tick = 0
            continue
        elif event.startswith('beat'):
            beat_tick = int(event.split('_')[1])
            shift = 0
        elif event.startswith('tempo'):
            tempo = int(event.split('_')[1])
            m = miditoolkit.midi.containers.TempoChange(time=bar_tick+beat_tick, tempo=tempo)
            midi.tempo_changes.append(m)
        elif event.startswith('shift'):
            shift = int(event.split('_')[1])
        elif event.startswith('pitch'):
            pitch = int(event.split('_')[1])
        elif event.startswith('velocity'):
            velocity = int(event.split('_')[1])
        elif event.startswith('duration'):
            duration = int(event.split('_')[1])
            n = miditoolkit.Note(
                start=bar_tick+beat_tick+shift,
                end=bar_tick+beat_tick+shift+duration,
                pitch=pitch,
                velocity=velocity
            )
            midi.instruments[0].notes.append(n)

    # print(song['notes'])
    # print(ticks, len(ticks))
    # print(list(map(lambda x: x % 120, ticks)))
    return midi

if __name__ == '__main__':
     # paths
    path_indir = './midi_analyzed'
    path_outdir = './corpus'
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    # n_files = 10
    print('num fiels:', n_files)

    # run all
    with mp.Pool() as pool:
        map_args = []
        for fidx in range(n_files):
            path_midi = midifiles[fidx]
            # print('{}/{}'.format(fidx, n_files))

            # paths
            path_infile = os.path.join(path_indir, path_midi)
            path_outfile = os.path.join(path_outdir, os.path.splitext(path_midi)[0])

            map_args.append((path_infile, path_outfile))

            # proc
            # proc_one(path_infile, path_outfile)

        for _ in tqdm(pool.imap(proc_one_map, map_args), total=n_files):
            pass
    
