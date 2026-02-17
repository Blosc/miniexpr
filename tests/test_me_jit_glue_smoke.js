'use strict';

const path = require('path');
const glue = require(path.join(__dirname, '..', 'src', 'me_jit_glue.js'));

function fail(message) {
    console.error(message);
    process.exit(1);
}

if (typeof glue._meJitInstantiate !== 'function') {
    fail('FAILED: _meJitInstantiate is not a function');
}
if (typeof glue._meJitFreeFn !== 'function') {
    fail('FAILED: _meJitFreeFn is not a function');
}

console.log('OK: me_jit_glue exports are present');
