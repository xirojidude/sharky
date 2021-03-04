
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class Teleport : UdonSharpBehaviour
{
    private VRCPlayerApi _playerLocal;
 //   public Transform target; //object being teleported to

    void Start()
    {
        _playerLocal = Networking.LocalPlayer;
    }


    // public override void OnPlayerTriggerEnter(VRCPlayerApi _playerLocal)
    // {
    //     _playerLocal.TeleportTo(target.position, target.localRotation);
    // }

    public void Interact() 
    {
        _playerLocal.TeleportTo(transform.position, transform.localRotation );
    }
}
