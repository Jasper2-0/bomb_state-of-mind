// Where the runtime data lives, relative to the page. Written by
// scripts/build-state-of-mind.sh: in a deployment the data sits beside the
// page, and in the repo it is one level up under work/. This is the only
// module whose content differs between what is reviewed and what is
// deployed -- everything else is byte-identical.
export const DATA_BASE = './data/';
