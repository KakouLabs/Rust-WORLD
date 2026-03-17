use rsworld_sys::{
    CheapTrick, CheapTrickOption, CodeAperiodicity, CodeSpectralEnvelope, D4COption,
    DecodeAperiodicity, DecodeSpectralEnvelope, Dio, DioOption, GetFFTSizeForCheapTrick,
    GetNumberOfAperiodicities, GetSamplesForDIO, GetSamplesForHarvest, Harvest, HarvestOption,
    StoneMask, Synthesis, D4C,
};

/// Build a row-pointer array from a flat row-major slice.
/// `rows` × `cols` must equal `data.len()`.
#[inline]
pub fn row_ptrs(data: &[f64], rows: usize, cols: usize) -> Vec<*const f64> {
    assert_eq!(
        data.len(),
        rows * cols,
        "row_ptrs: data.len() != rows * cols"
    );
    (0..rows).map(|r| data[r * cols..].as_ptr()).collect()
}

/// Build a mutable row-pointer array from a flat row-major slice.
/// `rows` × `cols` must equal `data.len()`.
#[inline]
pub fn row_ptrs_mut(data: &mut [f64], rows: usize, cols: usize) -> Vec<*mut f64> {
    assert_eq!(
        data.len(),
        rows * cols,
        "row_ptrs_mut: data.len() != rows * cols"
    );
    let base = data.as_mut_ptr();
    (0..rows).map(|r| unsafe { base.add(r * cols) }).collect()
}

pub fn cheaptrick(
    x: &[f64],
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    option: &mut CheapTrickOption,
) -> Vec<Vec<f64>> {
    let x_length: i32 = x.len() as i32;
    let f0_length: i32 = f0.len() as i32;
    unsafe {
        GetFFTSizeForCheapTrick(fs, option as *mut _);
    }
    let mut spectrogram = vec![vec![0.0; (option.fft_size / 2 + 1) as usize]; f0_length as usize];
    let mut spectrogram_ptr = spectrogram
        .iter_mut()
        .map(|inner| inner.as_mut_ptr())
        .collect::<Vec<_>>();
    let spectrogram_ptr = spectrogram_ptr.as_mut_ptr();
    unsafe {
        CheapTrick(
            x.as_ptr(),
            x_length,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_length,
            option as *const _,
            spectrogram_ptr,
        );
    }
    spectrogram
}

pub fn get_number_of_aperiodicities(fs: i32) -> i32 {
    unsafe { GetNumberOfAperiodicities(fs) }
}

pub fn code_aperiodicity(aperiodicity: &[Vec<f64>], f0_length: i32, fs: i32) -> Vec<Vec<f64>> {
    let mut cheaptrick_option = CheapTrickOption::new(fs);
    unsafe {
        GetFFTSizeForCheapTrick(fs, &mut cheaptrick_option as *mut _);
    }
    let fft_size = cheaptrick_option.fft_size;
    let aperiodicity_ptr = aperiodicity
        .iter()
        .map(|inner| inner.as_ptr())
        .collect::<Vec<_>>();
    let aperiodicity_ptr = aperiodicity_ptr.as_ptr();
    let n_aperiodicity;
    unsafe {
        n_aperiodicity = GetNumberOfAperiodicities(fs);
    }
    let mut coded_aperiodicity = vec![vec![0.0; n_aperiodicity as usize]; f0_length as usize];
    let mut coded_aperiodicity_ptr = coded_aperiodicity
        .iter_mut()
        .map(|inner| inner.as_mut_ptr())
        .collect::<Vec<_>>();
    let coded_aperiodicity_ptr = coded_aperiodicity_ptr.as_mut_ptr();
    unsafe {
        CodeAperiodicity(
            aperiodicity_ptr,
            f0_length,
            fs,
            fft_size,
            coded_aperiodicity_ptr,
        );
    }
    coded_aperiodicity
}

pub fn decode_aperiodicity(
    coded_aperiodicity: &[Vec<f64>],
    f0_length: i32,
    fs: i32,
) -> Vec<Vec<f64>> {
    let mut cheaptrick_option = CheapTrickOption::new(fs);
    unsafe {
        GetFFTSizeForCheapTrick(fs, &mut cheaptrick_option as *mut _);
    }
    let fft_size = cheaptrick_option.fft_size;
    let coded_aperiodicity_ptr = coded_aperiodicity
        .iter()
        .map(|inner| inner.as_ptr())
        .collect::<Vec<_>>();
    let coded_aperiodicity_ptr = coded_aperiodicity_ptr.as_ptr();
    let mut aperiodicity = vec![vec![0.0; (fft_size / 2 + 1) as usize]; f0_length as usize];
    let mut aperiodicity_ptr = aperiodicity
        .iter_mut()
        .map(|inner| inner.as_mut_ptr())
        .collect::<Vec<_>>();
    let aperiodicity_ptr = aperiodicity_ptr.as_mut_ptr();
    unsafe {
        DecodeAperiodicity(
            coded_aperiodicity_ptr,
            f0_length,
            fs,
            fft_size,
            aperiodicity_ptr,
        );
    }
    aperiodicity
}

pub fn code_spectral_envelope(
    spectrogram: &[Vec<f64>],
    f0_length: i32,
    fs: i32,
    fft_size: i32,
    number_of_dimensions: i32,
) -> Vec<Vec<f64>> {
    let spectrogram_ptr = spectrogram
        .iter()
        .map(|inner| inner.as_ptr())
        .collect::<Vec<_>>();
    let spectrogram_ptr = spectrogram_ptr.as_ptr();
    let mut coded_spectral_envelope =
        vec![vec![0.0; number_of_dimensions as usize]; f0_length as usize];
    let mut coded_spectral_envelope_ptr = coded_spectral_envelope
        .iter_mut()
        .map(|inner| inner.as_mut_ptr())
        .collect::<Vec<_>>();
    let coded_spectral_envelope_ptr = coded_spectral_envelope_ptr.as_mut_ptr();
    unsafe {
        CodeSpectralEnvelope(
            spectrogram_ptr,
            f0_length,
            fs,
            fft_size,
            number_of_dimensions,
            coded_spectral_envelope_ptr,
        );
    }
    coded_spectral_envelope
}

pub fn decode_spectral_envelope(
    coded_spectrogram: &[Vec<f64>],
    f0_length: i32,
    fs: i32,
    fft_size: i32,
) -> Vec<Vec<f64>> {
    let number_of_dimensions = coded_spectrogram[0].len() as i32;
    let mut spectrogram = vec![vec![0.0; (fft_size / 2 + 1) as usize]; f0_length as usize];
    let mut spectrogram_ptr = spectrogram
        .iter_mut()
        .map(|inner| inner.as_mut_ptr())
        .collect::<Vec<_>>();
    let spectrogram_ptr = spectrogram_ptr.as_mut_ptr();
    let coded_spectrogram_ptr = coded_spectrogram
        .iter()
        .map(|inner| inner.as_ptr())
        .collect::<Vec<_>>();
    let coded_spectrogram_ptr = coded_spectrogram_ptr.as_ptr();
    unsafe {
        DecodeSpectralEnvelope(
            coded_spectrogram_ptr,
            f0_length,
            fs,
            fft_size,
            number_of_dimensions,
            spectrogram_ptr,
        );
    }
    spectrogram
}

pub fn d4c(
    x: &[f64],
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    option: &D4COption,
) -> Vec<Vec<f64>> {
    let x_length = x.len() as i32;
    let f0_length = f0.len() as i32;
    let mut cheaptrick_option = CheapTrickOption::new(fs);
    unsafe {
        GetFFTSizeForCheapTrick(fs, &mut cheaptrick_option as *mut _);
    }
    let fft_size = cheaptrick_option.fft_size;
    let mut aperiodicity = vec![vec![0.0; (fft_size / 2 + 1) as usize]; f0_length as usize];
    let mut aperiodicity_ptr = aperiodicity
        .iter_mut()
        .map(|inner| inner.as_mut_ptr())
        .collect::<Vec<_>>();
    let aperiodicity_ptr = aperiodicity_ptr.as_mut_ptr();
    unsafe {
        D4C(
            x.as_ptr(),
            x_length,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_length,
            fft_size,
            option as *const _,
            aperiodicity_ptr,
        );
    }
    aperiodicity
}

pub fn dio(x: &[f64], fs: i32, option: &DioOption) -> (Vec<f64>, Vec<f64>) {
    let x_length = x.len() as i32;
    let f0_length: usize;
    unsafe {
        f0_length = GetSamplesForDIO(fs, x_length, option.frame_period) as usize;
    }
    let mut temporal_positions: Vec<f64> = vec![0.0; f0_length];
    let mut f0: Vec<f64> = vec![0.0; f0_length];
    unsafe {
        Dio(
            x.as_ptr(),
            x_length,
            fs,
            option as *const _,
            temporal_positions.as_mut_ptr(),
            f0.as_mut_ptr(),
        );
    }
    (temporal_positions, f0)
}

pub fn harvest(x: &[f64], fs: i32, option: &HarvestOption) -> (Vec<f64>, Vec<f64>) {
    let x_length = x.len() as i32;
    let f0_length: usize;
    unsafe {
        f0_length = GetSamplesForHarvest(fs, x_length, option.frame_period) as usize;
    }
    let mut temporal_positions: Vec<f64> = vec![0.0; f0_length];
    let mut f0: Vec<f64> = vec![0.0; f0_length];
    unsafe {
        Harvest(
            x.as_ptr(),
            x_length,
            fs,
            option as *const _,
            temporal_positions.as_mut_ptr(),
            f0.as_mut_ptr(),
        );
    }
    (temporal_positions, f0)
}

pub fn stonemask(x: &[f64], fs: i32, temporal_positions: &[f64], f0: &[f64]) -> Vec<f64> {
    let x_length = x.len() as i32;
    let f0_length = f0.len();
    let mut refined_f0 = vec![0.0; f0_length];
    unsafe {
        StoneMask(
            x.as_ptr(),
            x_length,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_length as i32,
            refined_f0.as_mut_ptr(),
        );
    }
    refined_f0
}

pub fn synthesis(
    f0: &[f64],
    spectrogram: &[Vec<f64>],
    aperiodicity: &[Vec<f64>],
    frame_period: f64,
    fs: i32,
) -> Vec<f64> {
    let f0_length = f0.len() as i32;
    let spectrogram_ptr = spectrogram
        .iter()
        .map(|inner| inner.as_ptr())
        .collect::<Vec<_>>();
    let spectrogram_ptr = spectrogram_ptr.as_ptr();
    let aperiodicity_ptr = aperiodicity
        .iter()
        .map(|inner| inner.as_ptr())
        .collect::<Vec<_>>();
    let aperiodicity_ptr = aperiodicity_ptr.as_ptr();
    let fft_size = (spectrogram[0].len() - 1) * 2;
    let y_length = f0_length * frame_period as i32 * fs / 1000;
    let mut y = vec![0.0; y_length as usize];
    unsafe {
        Synthesis(
            f0.as_ptr(),
            f0_length as i32,
            spectrogram_ptr,
            aperiodicity_ptr,
            fft_size as i32,
            frame_period,
            fs,
            y_length as i32,
            y.as_mut_ptr(),
        )
    }
    y
}

pub fn cheaptrick_flat(
    x: &[f64],
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    option: &mut CheapTrickOption,
) -> Vec<f64> {
    let x_length = x.len() as i32;
    let f0_length = f0.len() as i32;
    unsafe {
        GetFFTSizeForCheapTrick(fs, option as *mut _);
    }
    let cols = (option.fft_size / 2 + 1) as usize;
    let rows = f0_length as usize;
    let mut out = vec![0.0f64; rows * cols];
    let mut ptrs = row_ptrs_mut(&mut out, rows, cols);
    unsafe {
        CheapTrick(
            x.as_ptr(),
            x_length,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_length,
            option as *const _,
            ptrs.as_mut_ptr(),
        );
    }
    out
}

pub fn d4c_flat(
    x: &[f64],
    fs: i32,
    temporal_positions: &[f64],
    f0: &[f64],
    option: &D4COption,
) -> Vec<f64> {
    let x_length = x.len() as i32;
    let f0_length = f0.len() as i32;
    let mut cheaptrick_option = CheapTrickOption::new(fs);
    unsafe {
        GetFFTSizeForCheapTrick(fs, &mut cheaptrick_option as *mut _);
    }
    let fft_size = cheaptrick_option.fft_size;
    let cols = (fft_size / 2 + 1) as usize;
    let rows = f0_length as usize;
    let mut out = vec![0.0f64; rows * cols];
    let mut ptrs = row_ptrs_mut(&mut out, rows, cols);
    unsafe {
        D4C(
            x.as_ptr(),
            x_length,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_length,
            fft_size,
            option as *const _,
            ptrs.as_mut_ptr(),
        );
    }
    out
}

pub fn code_aperiodicity_flat(aperiodicity: &[f64], f0_length: i32, fs: i32) -> Vec<f64> {
    let mut cheaptrick_option = CheapTrickOption::new(fs);
    unsafe {
        GetFFTSizeForCheapTrick(fs, &mut cheaptrick_option as *mut _);
    }
    let fft_size = cheaptrick_option.fft_size;
    let rows = f0_length as usize;
    let cols_in = aperiodicity.len() / rows;
    let in_ptrs = row_ptrs(aperiodicity, rows, cols_in);
    let n_ap = unsafe { GetNumberOfAperiodicities(fs) } as usize;
    let mut out = vec![0.0f64; rows * n_ap];
    let mut out_ptrs = row_ptrs_mut(&mut out, rows, n_ap);
    unsafe {
        CodeAperiodicity(
            in_ptrs.as_ptr(),
            f0_length,
            fs,
            fft_size,
            out_ptrs.as_mut_ptr(),
        );
    }
    out
}

pub fn decode_aperiodicity_flat(coded_aperiodicity: &[f64], f0_length: i32, fs: i32) -> Vec<f64> {
    let mut cheaptrick_option = CheapTrickOption::new(fs);
    unsafe {
        GetFFTSizeForCheapTrick(fs, &mut cheaptrick_option as *mut _);
    }
    let fft_size = cheaptrick_option.fft_size;
    let rows = f0_length as usize;
    let cols_in = coded_aperiodicity.len() / rows;
    let in_ptrs = row_ptrs(coded_aperiodicity, rows, cols_in);
    let cols_out = (fft_size / 2 + 1) as usize;
    let mut out = vec![0.0f64; rows * cols_out];
    let mut out_ptrs = row_ptrs_mut(&mut out, rows, cols_out);
    unsafe {
        DecodeAperiodicity(
            in_ptrs.as_ptr(),
            f0_length,
            fs,
            fft_size,
            out_ptrs.as_mut_ptr(),
        );
    }
    out
}

pub fn code_spectral_envelope_flat(
    spectrogram: &[f64],
    f0_length: i32,
    fs: i32,
    fft_size: i32,
    number_of_dimensions: i32,
) -> Vec<f64> {
    let rows = f0_length as usize;
    let cols_in = (fft_size / 2 + 1) as usize;
    let in_ptrs = row_ptrs(spectrogram, rows, cols_in);
    let cols_out = number_of_dimensions as usize;
    let mut out = vec![0.0f64; rows * cols_out];
    let mut out_ptrs = row_ptrs_mut(&mut out, rows, cols_out);
    unsafe {
        CodeSpectralEnvelope(
            in_ptrs.as_ptr(),
            f0_length,
            fs,
            fft_size,
            number_of_dimensions,
            out_ptrs.as_mut_ptr(),
        );
    }
    out
}

pub fn decode_spectral_envelope_flat(
    coded_spectrogram: &[f64],
    f0_length: i32,
    fs: i32,
    fft_size: i32,
    number_of_dimensions: i32,
) -> Vec<f64> {
    let rows = f0_length as usize;
    let cols_in = coded_spectrogram.len() / rows;
    let in_ptrs = row_ptrs(coded_spectrogram, rows, cols_in);
    let cols_out = (fft_size / 2 + 1) as usize;
    let mut out = vec![0.0f64; rows * cols_out];
    let mut out_ptrs = row_ptrs_mut(&mut out, rows, cols_out);
    unsafe {
        DecodeSpectralEnvelope(
            in_ptrs.as_ptr(),
            f0_length,
            fs,
            fft_size,
            number_of_dimensions,
            out_ptrs.as_mut_ptr(),
        );
    }
    out
}

pub fn synthesis_flat(
    f0: &[f64],
    spectrogram: &[f64],
    aperiodicity: &[f64],
    fft_size: i32,
    frame_period: f64,
    fs: i32,
) -> Vec<f64> {
    let f0_length = f0.len() as i32;
    let spec_cols = (fft_size / 2 + 1) as usize;
    let rows = f0_length as usize;
    let spec_ptrs = row_ptrs(spectrogram, rows, spec_cols);
    let ap_ptrs = row_ptrs(aperiodicity, rows, spec_cols);
    let y_length = f0_length * frame_period as i32 * fs / 1000;
    let mut y = vec![0.0f64; y_length as usize];
    unsafe {
        Synthesis(
            f0.as_ptr(),
            f0_length,
            spec_ptrs.as_ptr(),
            ap_ptrs.as_ptr(),
            fft_size,
            frame_period,
            fs,
            y_length,
            y.as_mut_ptr(),
        );
    }
    y
}

#[cfg(test)]
mod tests {
    // CheapTrick test
    use crate::{cheaptrick, CheapTrickOption};

    #[test]
    fn test_cheaptrick() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let mut option = CheapTrickOption::new(fs);
        let spectrogram = cheaptrick(&x, fs, &temporal_positions, &f0, &mut option);
        assert_eq!(spectrogram.len(), f0.len());
        assert_eq!(spectrogram[0].len(), (option.fft_size / 2 + 1) as usize);
    }

    // Codec test
    use crate::{
        code_aperiodicity, code_spectral_envelope, decode_aperiodicity, decode_spectral_envelope,
        get_number_of_aperiodicities,
    };

    #[test]
    fn test_get_number_of_aperiodicities() {
        let fs = 44100;
        assert_eq!(get_number_of_aperiodicities(fs), 5);
    }

    #[test]
    fn test_code_aperiodicity() {
        let x = vec![0.0; 256];
        let fs = 44100 as i32;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let option = D4COption::new();

        let aperiodicity = d4c(&x, fs, &temporal_positions, &f0, &option);
        let coded_aperiodicity = code_aperiodicity(&aperiodicity, f0.len() as i32, fs);
        assert_eq!(coded_aperiodicity.len(), f0.len());
        assert_eq!(
            coded_aperiodicity[0].len(),
            get_number_of_aperiodicities(fs) as usize
        );
    }

    #[test]
    fn test_decode_aperiodicity() {
        let x = vec![0.0; 256];
        let fs = 44100 as i32;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let option = D4COption::new();

        let aperiodicity = d4c(&x, fs, &temporal_positions, &f0, &option);
        let coded_aperiodicity = code_aperiodicity(&aperiodicity, f0.len() as i32, fs);
        let decode_aperiodicity = decode_aperiodicity(&coded_aperiodicity, f0.len() as i32, fs);
        assert_eq!(decode_aperiodicity.len(), f0.len());
        assert_eq!(decode_aperiodicity[0].len(), (2048 / 2 + 1) as usize);
        assert_eq!(decode_aperiodicity[0][0], 0.999999999999);
    }

    #[test]
    fn test_code_spectral_envelope() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let mut option = CheapTrickOption::new(fs);
        let spectrogram = cheaptrick(&x, fs, &temporal_positions, &f0, &mut option);
        let number_of_dimensions = 256;
        let coded_spectrogram = code_spectral_envelope(
            &spectrogram,
            f0.len() as i32,
            fs,
            option.fft_size,
            number_of_dimensions,
        );
        assert_eq!(coded_spectrogram.len(), f0.len());
        assert_eq!(coded_spectrogram[0].len(), number_of_dimensions as usize);
    }

    #[test]
    fn test_decode_spectral_envelope() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let mut option = CheapTrickOption::new(fs);
        let spectrogram = cheaptrick(&x, fs, &temporal_positions, &f0, &mut option);
        let number_of_dimensions = 256;
        let coded_spectrogram = code_spectral_envelope(
            &spectrogram,
            f0.len() as i32,
            fs,
            option.fft_size,
            number_of_dimensions,
        );
        let spectrogram =
            decode_spectral_envelope(&coded_spectrogram, f0.len() as i32, fs, option.fft_size);
        assert_eq!(spectrogram.len(), f0.len());
        assert_eq!(spectrogram[0].len(), (option.fft_size / 2 + 1) as usize);
    }

    // D4C test
    use crate::{d4c, D4COption};

    #[test]
    fn test_d4c() {
        let x = vec![0.0; 256];
        let fs = 44100 as i32;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let option = D4COption::new();
        let aperiodicity = d4c(&x, fs, &temporal_positions, &f0, &option);
        assert_eq!(aperiodicity.len(), f0.len());
        assert_eq!(aperiodicity[0].len(), (2048 / 2 + 1) as usize);
        assert_eq!(aperiodicity[0][0], 0.999999999999);
    }

    // DIO test
    use crate::{dio, DioOption};

    #[test]
    fn test_dio() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let option = DioOption::new();
        let (temporal_positions, f0) = dio(&x, fs, &option);
        assert_eq!(temporal_positions, vec![0.0, 0.005]);
        assert_eq!(f0, vec![0.0, 0.0]);
    }

    // Harvest test
    use crate::{harvest, HarvestOption};

    #[test]
    fn test_harvest() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let option = HarvestOption::new();
        let (temporal_positions, f0) = harvest(&x, fs, &option);
        assert_eq!(temporal_positions, vec![0.0, 0.005]);
        assert_eq!(f0, vec![0.0, 0.0]);
    }

    // StoneMask test
    use crate::stonemask;

    #[test]
    fn test_stonemask() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let option = DioOption::new();
        let (temporal_positions, f0) = dio(&x, fs, &option);
        let refined_f0 = stonemask(&x, fs, &temporal_positions, &f0);
        assert_eq!(refined_f0, vec![0.0, 0.0]);
    }

    // Synthesis test
    use crate::synthesis;

    #[test]
    fn test_synthesis() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let option = DioOption::new();
        let (temporal_positions, f0) = dio(&x, fs, &option);
        let frame_period = option.frame_period;
        let f0 = stonemask(&x, fs, &temporal_positions, &f0);
        let mut option = CheapTrickOption::new(fs);
        let spectrogram = cheaptrick(&x, fs, &temporal_positions, &f0, &mut option);
        let option = D4COption::new();
        let aperiodicity = d4c(&x, fs, &temporal_positions, &f0, &option);
        let y = synthesis(&f0, &spectrogram, &aperiodicity, frame_period, fs);
        let y_length = f0.len() as i32 * frame_period as i32 * fs / 1000;
        assert_eq!(y.len(), y_length as usize);
    }

    use crate::{
        cheaptrick_flat, code_aperiodicity_flat, code_spectral_envelope_flat, d4c_flat,
        decode_aperiodicity_flat, decode_spectral_envelope_flat, synthesis_flat,
    };

    #[test]
    fn test_cheaptrick_flat() {
        let x: Vec<f64> = vec![0.0; 256];
        let fs = 44100;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let mut option = CheapTrickOption::new(fs);
        let flat = cheaptrick_flat(&x, fs, &temporal_positions, &f0, &mut option);
        let cols = (option.fft_size / 2 + 1) as usize;
        assert_eq!(flat.len(), f0.len() * cols);
    }

    #[test]
    fn test_d4c_flat() {
        let x = vec![0.0; 256];
        let fs = 44100_i32;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let option = D4COption::new();
        let flat = d4c_flat(&x, fs, &temporal_positions, &f0, &option);
        let cols = (2048 / 2 + 1) as usize;
        assert_eq!(flat.len(), f0.len() * cols);
        assert_eq!(flat[0], 0.999999999999);
    }

    #[test]
    fn test_code_aperiodicity_flat() {
        let x = vec![0.0; 256];
        let fs = 44100_i32;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let option = D4COption::new();
        let ap_flat = d4c_flat(&x, fs, &temporal_positions, &f0, &option);
        let coded = code_aperiodicity_flat(&ap_flat, f0.len() as i32, fs);
        let n_ap = get_number_of_aperiodicities(fs) as usize;
        assert_eq!(coded.len(), f0.len() * n_ap);
    }

    #[test]
    fn test_decode_aperiodicity_flat() {
        let x = vec![0.0; 256];
        let fs = 44100_i32;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let option = D4COption::new();
        let ap_flat = d4c_flat(&x, fs, &temporal_positions, &f0, &option);
        let coded = code_aperiodicity_flat(&ap_flat, f0.len() as i32, fs);
        let decoded = decode_aperiodicity_flat(&coded, f0.len() as i32, fs);
        let cols_in = (2048 / 2 + 1) as usize;
        assert_eq!(decoded.len(), f0.len() * cols_in);
        assert_eq!(decoded[0], 0.999999999999);
    }

    #[test]
    fn test_code_spectral_envelope_flat() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let mut option = CheapTrickOption::new(fs);
        let spec_flat = cheaptrick_flat(&x, fs, &temporal_positions, &f0, &mut option);
        let number_of_dimensions = 256_i32;
        let coded = code_spectral_envelope_flat(
            &spec_flat,
            f0.len() as i32,
            fs,
            option.fft_size,
            number_of_dimensions,
        );
        assert_eq!(coded.len(), f0.len() * number_of_dimensions as usize);
    }

    #[test]
    fn test_decode_spectral_envelope_flat() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let temporal_positions = vec![0.0, 0.005];
        let f0 = vec![0.0, 0.0];
        let mut option = CheapTrickOption::new(fs);
        let spec_flat = cheaptrick_flat(&x, fs, &temporal_positions, &f0, &mut option);
        let cols_in = (option.fft_size / 2 + 1) as usize;
        let number_of_dimensions = 256_i32;
        let coded = code_spectral_envelope_flat(
            &spec_flat,
            f0.len() as i32,
            fs,
            option.fft_size,
            number_of_dimensions,
        );
        let decoded = decode_spectral_envelope_flat(
            &coded,
            f0.len() as i32,
            fs,
            option.fft_size,
            number_of_dimensions,
        );
        assert_eq!(decoded.len(), f0.len() * cols_in);
    }

    #[test]
    fn test_synthesis_flat() {
        let x = vec![0.0; 256];
        let fs = 44100;
        let option = DioOption::new();
        let (temporal_positions, f0) = dio(&x, fs, &option);
        let frame_period = option.frame_period;
        let f0 = stonemask(&x, fs, &temporal_positions, &f0);
        let mut ct_option = CheapTrickOption::new(fs);
        let spec_flat = cheaptrick_flat(&x, fs, &temporal_positions, &f0, &mut ct_option);
        let d4c_option = D4COption::new();
        let ap_flat = d4c_flat(&x, fs, &temporal_positions, &f0, &d4c_option);
        let y = synthesis_flat(
            &f0,
            &spec_flat,
            &ap_flat,
            ct_option.fft_size,
            frame_period,
            fs,
        );
        let y_length = f0.len() as i32 * frame_period as i32 * fs / 1000;
        assert_eq!(y.len(), y_length as usize);
    }
}
