use crate::{
    node::{AudioNodeChannelConfig, ParamData, ParamID, ParamInfo, ParamUnit, ParamUpdate},
    BlockBuffer, BlockFrames, NormalVal, MAX_BLOCK_SIZE,
};

pub struct TestToneGeneratorNode;

impl crate::node::AudioNode for TestToneGeneratorNode {
    fn parameters(&self) -> Vec<ParamInfo> {
        vec![
            ParamInfo {
                id: 0,
                name: "Volume".into(),
                num_steps: None,
                unit: ParamUnit::Decibels,
                default: NormalVal::new(0.5),
            },
            ParamInfo {
                id: 1,
                name: "Frequency".into(),
                num_steps: None,
                unit: ParamUnit::FreqHz,
                default: NormalVal::new(0.5),
            },
        ]
    }

    fn channel_config(&self) -> crate::node::AudioNodeChannelConfig {
        AudioNodeChannelConfig {
            num_inputs: 0,
            num_outputs: 2,
        }
    }

    fn activate(
        &mut self,
        _instance_id: u32,
        info: crate::node::ActiveServerInfo,
        initial_params: &[(ParamID, ParamData)],
    ) -> Box<dyn crate::node::AudioNodeProcessor> {
        let mut volume_norm = NormalVal::new(0.5);
        let mut freq_norm = NormalVal::new(0.5);

        for (param_id, param_data) in initial_params.iter() {
            match *param_id {
                0 => {
                    if let ParamData::Normal(n) = param_data {
                        volume_norm = *n;
                    }
                }
                1 => {
                    if let ParamData::Normal(n) = param_data {
                        freq_norm = *n;
                    }
                }
                _ => {}
            }
        }

        Box::new(Processor {
            freq_hz: 440.0,
            amplitude_raw: 0.25,

            phasor: 0.0,
            phasor_inc: 440.0 / info.sample_rate as f32,
        })
    }
}

struct Processor {
    freq_hz: f32,
    amplitude_raw: f32,

    phasor: f32,
    phasor_inc: f32,
}

impl Processor {
    fn update_param(&mut self, update: &ParamUpdate) {
        match update.id {
            0 => {}
            1 => {}
            _ => {}
        }
    }
}

impl crate::node::AudioNodeProcessor for Processor {
    fn process(
        &mut self,
        _cx: &crate::node::ProcessContext,
        frames: BlockFrames<MAX_BLOCK_SIZE>,
        param_updates: &[ParamUpdate],
        _inputs: &[&BlockBuffer<MAX_BLOCK_SIZE>],
        outputs: &mut [&mut BlockBuffer<MAX_BLOCK_SIZE>],
    ) {
        let (out_buf_l, out_buf_r) = crate::util::output_stereo(outputs);

        let mut is_silent = self.amplitude_raw == 0.0;

        crate::util::param_update_chunks(frames, param_updates, |param_updates, range| {
            for update in param_updates.iter() {
                self.update_param(update);
            }

            let out_l = out_buf_l.range_mut(range.clone());
            let out_r = out_buf_r.range_mut(range);

            if self.amplitude_raw == 0.0 {
                out_l.fill(0.0);
                out_r.fill(0.0);
            } else {
                is_silent = false;

                for (l, r) in out_l.iter_mut().zip(out_r.iter_mut()) {
                    let val = (self.phasor * std::f32::consts::TAU).sin() * self.amplitude_raw;
                    self.phasor = (self.phasor + self.phasor_inc).fract();

                    *l = val;
                    *r = val;
                }
            }
        });

        out_buf_l.is_silent = is_silent;
        out_buf_r.is_silent = is_silent;
    }
}
