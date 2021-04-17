using UdonSharp;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using VRC.SDK3.Components;
using VRC.SDK3.Components.Video;
using VRC.SDKBase;
using VRC.Udon;

public class SkyItem : UdonSharpBehaviour
{
    public int id ;
    public Material shader;
    public string label ;
    public string cat;
    public int gpu ;
    public float ground ;
    public int sun;
    public string video ;
    public VRCUrl videoURL;
    public int enabled;
}
