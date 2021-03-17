
using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class Joystick_01 : UdonSharpBehaviour
{

    //[SerializeField] Transform Ship;
    public Transform Ship;
    private float speed = 0.0f;
    private VRCPlayerApi player;
    private float smooth = 1.1f;

    void Start()
    {
        player = Networking.LocalPlayer;
    }
    void Update() {

        if (speed >0.0f) 
        {
        Ship.position += transform.forward * Time.deltaTime * speed;
        Quaternion newDirection = player.GetBoneRotation( HumanBodyBones.Head );
        newDirection *= Quaternion.Euler(0, -90, 0);
        Vector3 angles = newDirection.eulerAngles;
        newDirection = Quaternion.Euler(angles.x, angles.y, -angles.z);
        Ship.rotation = (Quaternion.Slerp(Ship.rotation, newDirection,  Time.deltaTime * smooth));
        //Ship.Rotate( Input.GetAxis("Vertical"), 0.0f, -Input.GetAxis("Horizontal") );
        }
    }

    void OnPickupUseDown()
    {
        speed = 40.0f;
    }

    void OnPickupUseUp()
    {
        speed = 0.0f;
    }

    void OnPickup()
    {
        Ship.position += transform.forward * Time.deltaTime * 40.0f;

        Ship.Rotate( Input.GetAxis("Vertical"), 0.0f, -Input.GetAxis("Horizontal") );

//        float terrainHeightWhereWeAre = Terrain.activeTerrain.SampleHeight( transform.position );

//        if (terrainHeightWhereWeAre > transform.position.y) {
//        transform.position = new Vector3(transform.position.x,
//        terrainHeightWhereWeAre,
//        transform.position.z);
    }

    void OnDrop()
    {
        // reset position of joystick to original
    }
}
