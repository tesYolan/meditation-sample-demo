from pydub import AudioSegment

def merge_audios(main_track, background_track, output_file):
    main_audio = AudioSegment.from_file(main_track)
    background_audio = AudioSegment.from_file(background_track, format="mp4")
    print("lenghts are ")
    print(len(main_audio))
    print(len(background_audio))

    background_audio = background_audio * (len(main_audio) // len(background_audio) + 1)
    print(len(background_audio))

    background_audio = background_audio[:len(main_audio)]
    print(len(background_audio))

    # Overlay main and background
    output_audio = background_audio.overlay(main_audio)

    # Export merged audio file
    output_audio.export(output_file, format="wav")

    return output_file

def generate_playable_thumbnail(audio_file, thumbnail_file, output_file):
    pass