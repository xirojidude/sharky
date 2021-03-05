
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class SkyButton : UdonSharpBehaviour
{
    public Material skybox;

    public override void Interact()
    {
        RenderSettings.skybox = skybox;
    }
}
