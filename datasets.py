from cmsmusic import Dataset, DatasetType, LHCRun, NanoADODVersion, ProcessGroup, Year

datasets: list[Dataset] = []

datasets.append(
    Dataset(
        das_names="/DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/RunIII2024Summer24NanoAODv15-150X_mcRun3_2024_realistic_v2-v6/NANOAODSIM",
        process_group=ProcessGroup.DRELL_YAN,
        year=Year.RunSummer24,
        nanoadod_version=NanoADODVersion.V15,
        lhc_run=LHCRun.Run3,
        dataset_type=DatasetType.BACKGROUND,
        xsec=1.0,
        filter_eff=1.0,
        k_factor=1.0,
    )
)
datasets.append(
    Dataset(
        das_names=[
            "/Muon0/Run2024G-MINIv6NANOv15-v1/NANOAOD",
        ],
        process_group=ProcessGroup.DATA,
        year=Year.RunSummer24,
        nanoadod_version=NanoADODVersion.V15,
        lhc_run=LHCRun.Run3,
        dataset_type=DatasetType.DATA,
        xsec=1.0,
        filter_eff=1.0,
        k_factor=1.0,
    )
)
datasets.append(
    Dataset(
        das_names=[
            "/Muon1/Run2024G-MINIv6NANOv15-v2/NANOAOD",
        ],
        process_group=ProcessGroup.DATA,
        year=Year.RunSummer24,
        nanoadod_version=NanoADODVersion.V15,
        lhc_run=LHCRun.Run3,
        dataset_type=DatasetType.DATA,
        xsec=1.0,
        filter_eff=1.0,
        k_factor=1.0,
    )
)
