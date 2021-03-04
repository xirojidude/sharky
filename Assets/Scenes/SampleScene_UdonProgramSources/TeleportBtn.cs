using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class TeleportBtn : UdonSharpBehaviour
{
//  public GameObject Target;

//     void Interact()
//     {
//         Target.SetActive(!Target.activeInHierarchy);
//     }
[SerializeField] Transform target;

    public override void Interact()
    {
        Networking.LocalPlayer.TeleportTo(target.position, target.rotation);
    }
}
