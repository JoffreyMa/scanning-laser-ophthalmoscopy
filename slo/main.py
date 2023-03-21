from evaluate import evaluate_process
from process import apply_denoise_gaussian_canny, apply_sato_hysteresis, apply_background_removal_meijering, apply_background_removal_sato, apply_background_removal_black_tophat, apply_thick_thin, apply_mix_or

if __name__ == '__main__':
    evaluate_process(process=apply_denoise_gaussian_canny)
    evaluate_process(process=apply_sato_hysteresis)
    evaluate_process(process=apply_background_removal_meijering)
    evaluate_process(process=apply_background_removal_sato)
    evaluate_process(process=apply_background_removal_black_tophat)
    evaluate_process(process=apply_thick_thin)