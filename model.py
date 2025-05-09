from transformers import WavLMForCTC, AutoFeatureExtractor, AutoProcessor, Wav2Vec2ForCTC, Wav2Vec2Processor, DataCollator
from torchaudio.transforms import PadTrim
import torch


class WavLM:
    def __init__(self, audio_path, threshold=0.5, unfreeze_model=False):
        self.audio = audio
        self.threshold = threshold
        self.model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        self.feature_extractor = AutoProcessor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
        self.unfreeze_model = unfreeze_model

    def extract_probs(self):
        inputs = self.feature_extractor(self.audio, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values
        with torch.no_grad():
            logits = self.model(input_values).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def speaker_separation(self, n_speakers=2):
        probs = self.extract_probs()
        # Get the probabilities of each speaker
        speaker_segments = (probs[:, :, (n_speakers-1)] > self.threshold).int()

        # Apply simple clustering
        current_speaker = speaker_segments[0, 0].item()
        speaker_segments_list = []
        start_idx = 0

        for i in range(1, len(speaker_segments[0])):
            if speaker_segments[0, i].item() != current_speaker:
                speaker_segments_list.append((start_idx, i - 1, current_speaker))
                start_idx = i
                current_speaker = speaker_segments[0, i].item()

            # Add the last segment
        speaker_segments_list.append((start_idx, len(speaker_segments[0]) - 1, current_speaker))

        # Print the identified speaker segments
        for segment in speaker_segments_list:
            start, end, speaker_id = segment
            print(f"Speaker {speaker_id} speaks from {start} to {end} seconds.")


class SpeakerSeparationCollator(DataCollator):
    def _init_(self, max_waveform_length):
        self.pad_trim = PadTrim(max_waveform_length)

    def collate_batch(self, batch):
        sampling_rates, mixture_waveforms, source_waveforms_list = zip(*batch)

        # Your preprocessing steps go here (if needed)
        # ...

        # Use PadTrim to make the waveforms have the same size
        mixture_waveforms = [self.pad_trim(torch.tensor(waveform)) for waveform in mixture_waveforms]
        source_waveforms_list = [[self.pad_trim(torch.tensor(source)) for source in sources] for sources in
                                 source_waveforms_list]

        # Stack the padded/truncated waveforms
        mixture_waveforms = torch.stack(mixture_waveforms)
        source_waveforms_list = [torch.stack(sources) for sources in source_waveforms_list]

        return {
            'sampling_rates': sampling_rates,
            'input_values': mixture_waveforms,  # Adjust this based on your model's input requirements
            'labels': source_waveforms_list,
        }
# class Wav2Vec2:
#     def __init__(self, audio_path, threshold=0.5, unfreeze_model=False):
#         self.audio = audio
#         self.threshold = threshold
#         self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
#         self.feature_extractor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
#         self.unfreeze_model = unfreeze_model
#
#     def