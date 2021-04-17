
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class SendEventTest : UdonSharpBehaviour
{ 
    public int _newSky=0;
    [Tooltip("This list of skyboxes for the world")]
    public Material[] skybox;


    public void NextSky() 
    {
        Debug.Log("******************  NextSky Triggered  ***********************");

       _newSky++;
       if (_newSky>=skybox.Length) _newSky=0;

       RenderSettings.skybox = skybox[_newSky];
    }
}
