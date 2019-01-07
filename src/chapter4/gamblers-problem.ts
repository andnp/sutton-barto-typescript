import { arrays } from 'utilities-ts';
import * as _ from 'lodash';

import { LineChart, Remote } from 'tsplot';
import { argMax } from '../utils/numerical';

const theta = 0.00001; // precision parameter
const ph = 0.55; // probability of heads
const max_steps = 100; // extra termination condition in case convergence fails
const gamma = 1;

const states = arrays.range(100);

async function exec() {
    let delta = 0;
    let steps = 0;

    const value = _.times(101, i => 0);

    while ((delta > theta || steps < 1) && steps < max_steps) {
        steps++;
        delta = 0;
        for (const s of states) {
            const old_v = value[s];
            const potential_actions = arrays.range(1, s + 1);

            const potential_outcomes = potential_actions.map(a => {
                return getExpectedReturn(s, a, value);
            });

            const max_value = Math.max(...potential_outcomes, 0);
            value[s] = max_value;

            delta = Math.max(delta, Math.abs(old_v - max_value));
        }

        console.log(steps, delta);
    }

    const value_plot = LineChart.fromArray(value.slice(0, -1));

    const policy = states.map(s => {
        const potential_actions = arrays.range(1, s + 1);
        const potential_values = potential_actions.map(a => {
            return getExpectedReturn(s, a, value);
        });

        const a_idx = argMax(potential_values, { breakTieRandomly: false });
        return potential_actions[a_idx];
    });

    const policy_plot = LineChart.fromArray(policy.slice(0, -1));
    policy_plot.smooth(false);

    await Remote.plot([value_plot, policy_plot]);
}

exec().catch(console.log);

function getExpectedReturn(s: number, a: number, value: number[]) {
    const new_state_success = s + a > 100 ? 100 : s + a;
    const new_state_fail = s - a;
    const estimated_return_success = getReward(new_state_success) + gamma * value[new_state_success];
    const estimated_return_fail = getReward(new_state_fail) + gamma * value[new_state_fail];

    return ph * estimated_return_success + (1 - ph) * estimated_return_fail;
}

function getReward(state: number) {
    return state >= 100 ? 1 : 0;
}
