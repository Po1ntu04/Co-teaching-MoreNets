# Copy this file to config.psd1 and fill in your real values.
@{
    SshHostAlias       = "b101"
    Host               = "b101.guhk.cc"
    Port               = 22
    User               = "yuzhixiang"
    RepoParentDir      = "/data1/yuzhixiang/work"
    RepoDir            = "/data1/yuzhixiang/work/Co-teaching-MoreNets"
    GitRemote          = "origin"
    LocalGitUrl        = "git@github.com:Po1ntu04/Co-teaching-MoreNets.git"
    RemoteGitUrl       = "https://github.com/Po1ntu04/Co-teaching-MoreNets.git"
    CondaInit          = "source /data1/yuzhixiang/opt/miniconda3/etc/profile.d/conda.sh"
    DefaultCondaEnv    = "/data1/yuzhixiang/.conda/envs/coteaching-py39"
    DefaultTmuxSession = "coteaching-run"
    DefaultTmuxLog     = "logs/workflow/tmux_run.log"
    DefaultResultsDir  = "results"
}
