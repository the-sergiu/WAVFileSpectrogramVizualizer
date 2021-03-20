clc;
clear all;
close all;

[perfectSound, freq] = audioread('piano_short.wav');

noisySound = perfectSound + 0.01 * randn(length(perfectSound), 1);

promptMessage = sprintf('Which sound do you want to hear?');
titleBarCaption = 'Specify Sound';
button = questdlg(promptMessage, titleBarCaption, 'Perfect', 'Noisy', 'Perfect');

if strcmpi(button, 'Perfect')
  soundsc(perfectSound, freq);
else
  audiowrite('piano_short_noisy.wav',noisySound, freq);
  soundsc(noisySound, freq);
end







