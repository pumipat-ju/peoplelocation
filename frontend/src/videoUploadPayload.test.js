import test from 'node:test';
import assert from 'node:assert/strict';

import { appendVideoUploadFields } from './videoUploadPayload.js';


test('configured video offset is included in the upload form payload', () => {
  const entries = [];
  const formData = {
    append(name, value) {
      entries.push([name, value]);
    },
  };
  const file = { name: 'camera-a.mp4' };

  appendVideoUploadFields(formData, { file, offset: '-1.25' }, 'camera-a');

  assert.deepEqual(entries, [
    ['name', 'camera-a'],
    ['file', file],
    ['loop_video', 'true'],
    ['time_offset_sec', '-1.25'],
  ]);
});
