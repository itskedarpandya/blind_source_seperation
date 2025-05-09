from utils import LibriMixDataset, PlotLibriMix
#from model import WavLM, SpeakerSeparationCollator
from transformers import DataCollatorWithPadding



train_set, val_set = LibriMixDataset(dev_mode=True).load_dataset()

#train_set = SpeakerSeparationCollator(train_set).collate_batch()
sampling_rates, mixture_waveforms, source_waveforms_list = zip(*train_set)
max_waveform_length = max(len(waveform[0]) for waveform in mixture_waveforms)
print(max_waveform_length)
# for waveform in mixture_waveforms:
#     li = []
#     li = li.append(max(waveform.shape))
#     print(max(li))

#PlotLibriMix(train_set, 0).plot_graph()
# wav_lm = WavLM(train_set, threshold=0.5)
# wav_lm.speaker_separation(n_speakers=2)