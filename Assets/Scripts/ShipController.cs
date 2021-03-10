
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class ShipController : UdonSharpBehaviour
{
    private Vector3 speed = new Vector3(0,0,0);
    private Vector3 angularSpeed = new Vector3(0,0,0);
    public Vector3 joystick = new Vector3(0,0,0);
    private Vector3 maxAcceleration = new Vector3(0,0,0);
    private Vector3 maxRotate = new Vector3(0,0,0);
    public Vector3 drag = new Vector3(0,0,0);
    public Vector3 angularDrag = new Vector3(0,0,0);

    private VRCPlayerApi playerLocal;
    private bool isActive;

    void Start()
    {
        playerLocal = Networking.LocalPlayer;
    }

    private void onPickupUseDown() 
    {
        isActive = true;
    }

    private void onPickupUseUp()
    {
        isActive = false;
    }

    private void fixedUpdate()
    {
        if(isActive) 
        {
            playerLocal.SetVelocity(Vector3.ClampMagnitude(playerLocal.GetVelocity() + transform.forward, 50));
        }
    }
}
