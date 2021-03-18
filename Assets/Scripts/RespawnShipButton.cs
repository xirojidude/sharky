
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class RespawnShipButton : UdonSharpBehaviour
{
    public GameObject Target;
    private Vector3 originalPostion;
    private Quaternion originalRotation;

    void Start()
    {
        originalPostion = Target.transform.position;    
    }

      void Interact()
    {
        Target.transform.position=originalPostion;
        Target.transform.rotation=originalRotation;
    }
}
