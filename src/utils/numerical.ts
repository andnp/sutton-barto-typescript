import { random } from "mlts";

interface ArgMaxOptions {
    breakTieRandomly?: boolean;
}

export const argMax = (arr: number[], opts?: ArgMaxOptions): number => {
    let m = Number.NEGATIVE_INFINITY;
    let locs = [] as number[];
    for (let i = 0; i < arr.length; ++i) {
        if (arr[i] === m) {
            locs.push(i);
        } else if (arr[i] > m) {
            locs = [i];
            m = arr[i];
        }
    }

    const breakTieRandomly = !opts || (opts && opts.breakTieRandomly);
    if (locs.length < 2 || !breakTieRandomly) return locs[0];
    const rnd = random.randomInteger(0, locs.length - 1);
    return locs[rnd];
};
