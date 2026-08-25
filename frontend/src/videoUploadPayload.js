export function appendVideoUploadFields(formData, fileObj, cameraName) {
  formData.append('name', cameraName);
  formData.append('file', fileObj.file);
  formData.append('loop_video', 'true');
  formData.append('time_offset_sec', String(fileObj.offset || 0));
  return formData;
}
