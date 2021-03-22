
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.SDK3.Components;
using VRC.SDK3.Components.Video;
using VRC.SDK3.Video.Components.Base;
using VRC.Udon;

public class VRCWorld : UdonSharpBehaviour
{
    private VRCPlayerApi player;
    public  BaseVRCVideoPlayer  videoPlayer;
    public  VRCUrlInputField    videoURLInputField;
    private VRCUrl url = VRCUrl.Empty;



    void Start()
    {
        player = Networking.LocalPlayer;
//        videoPlayer = (BaseVRCVideoPlayer) GetComponent(typeof(BaseVRCVideoPlayer));

    }

    void OnPlayerJoined(VRCPlayerApi newPlayer)
    {
        Debug.Log($"[Player Joined] {player.displayName} arrived at SkyTower");
//        videoPlayer.Stop();
//        url = VRCUrl.Empty;
//        videoURLInputField.SetUrl($"https://vrcstreamtest.herokuapp.com/?user={player.displayName}");    // .url = $"https://vrcstreamtest.herokuapp.com/?user={player.displayName}";
//        videoPlayer.LoadURL(videoURLInputField.GetUrl());
    }
}
