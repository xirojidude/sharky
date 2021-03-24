
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.SDK3.Components.Video;
using VRC.SDK3.Video.Components;
using VRC.SDK3.Video.Components.AVPro;
using VRC.SDK3.Video.Components.Base;
using VRC.Udon;

public class SkyButton : UdonSharpBehaviour
{
    public Material skybox;
    public VRCAVProVideoPlayer videoPlayer;
    public VRCUrl  soundUrl;

    public override void Interact()
    {
        RenderSettings.skybox = skybox;
        if (videoPlayer!=null) {
            videoPlayer.LoadURL(soundUrl);
            //videoPlayer.play();
        }
    }
}
