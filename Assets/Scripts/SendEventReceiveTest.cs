
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class SendEventReceiveTest : UdonSharpBehaviour
{
 //   public int _syncedSky=0;
 //    [Tooltip("This list of skyboxes for the world")]
 //   public Material[] skybox;


    void Start()
    {
        
    }

    public void NextSky() 
    {
        Debug.Log("******************  NextSky Triggered  ***********************");
//        GameObject obj = this.GetComponent<GameObject>();
//        obj.SetActive(!obj.activeInHierarchy);
//        _syncedSky++;
//        if (_syncedSky>=skybox.Length) _syncedSky=0;

//        RenderSettings.skybox = skybox[_syncedSky];
    }
}
