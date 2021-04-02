
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class PilotSeat : UdonSharpBehaviour
{
    public GameObject Joystick;
    private Vector3 originalPostion;
    private Quaternion originalRotation;

    void Start()
    {
        
    }
    void OnStationEntered() 
    {
        originalPostion = Joystick.transform.localPosition;
        originalRotation = Joystick.transform.localRotation;
 
    }
    void OnStationExited() {
        Joystick.transform.localPosition = originalPostion;
        Joystick.transform.localRotation = originalRotation;
    }
}
