clear all
addpath ../scatnet-0.2a
addpath_scatnet;
addpath ../minFunc_2012/minFunc
addpath ../minFunc_2012/minFunc/compiled

%name = 'tur2a';
%name = 'anisotur2a';
%name = 'mrw2dd';
name = 'bubbles';
for ks = 1 : 1 % 0
    maxent_modelA(name,ks);
end
