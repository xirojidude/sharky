
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class BtnToggle : UdonSharpBehaviour
{
 public GameObject Target;

    void Interact()
    {
        Target.SetActive(!Target.activeInHierarchy);
    }
}
