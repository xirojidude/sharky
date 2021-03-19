
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class RespawnShipButton : UdonSharpBehaviour
{
    public GameObject Target;
    public GameObject SpawnLocation;
    private Vector3 originalPostion;
    private Quaternion originalRotation;

    void Start()
    {
        originalPostion = Target.transform.position;    
        //originalPostion = new Vector3(-7.2329f,2.19f,-1.083f);
        originalRotation = Target.transform.rotation;
        //7.2329,2.19,-1.083
    }

      void Interact()
    {
//        Target.transform.position=originalPostion;
//        Target.transform.rotation=originalRotation;
        Target.transform.position=SpawnLocation.transform.position;
        Target.transform.rotation=SpawnLocation.transform.rotation;
    }
}
