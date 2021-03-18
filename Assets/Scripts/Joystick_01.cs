﻿
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
    private Vector3 joystickSpawnPosition;
    private Quaternion joystickSpawnRotation;

    void Start()
    {
        player = Networking.LocalPlayer;
        joystickSpawnPosition = this.transform.localPosition;
        joystickSpawnRotation = this.transform.localRotation;
    }
    void Update() {

        if (speed >0.0f) 
        {
        Ship.position += transform.forward * Time.deltaTime * speed;

//  From a quaternion ...
//  roll  = Mathf.Atan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z);
//  pitch = Mathf.Atan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z);
//  yaw   =  Mathf.Asin(2*x*y + 2*z*w);

        float pitch = (transform.rotation * Quaternion.Euler(180, 0, 0)).eulerAngles.x;
        float roll = 180-(transform.rotation * Quaternion.Euler(0, 0, 180)).eulerAngles.z;
        Debug.Log("Pitch:" + pitch + "  Roll:" + roll + "  Heading:" + transform.rotation.y + ")");

        // Quaternion bodyDirection =player.GetRotation();
        // Vector3 bodyVector = bodyDirection.eulerAngles;
        // Debug.Log("body(" + bodyVector.x + ", " + bodyVector.y + ", " + bodyVector.z + ")");

        Vector3 shipVector = Ship.rotation.eulerAngles;
        Debug.Log("ship(" + Ship.rotation.eulerAngles.x + ", " + Ship.rotation.eulerAngles.y + ", " + Ship.rotation.eulerAngles.z + ") LocalTransforn(" + transform.localEulerAngles.x + ", " + transform.localEulerAngles.y + ", " + transform.localEulerAngles.z + ")");

        // Quaternion headDirection = player.GetBoneRotation( HumanBodyBones.Head );
        // Vector3 headVector = headDirection.eulerAngles;
        // Debug.Log("head(" + headVector.x + ", " + headVector.y + ", " + headVector.z + ")  Transforn(" + transform.rotation.x + ", " + transform.rotation.y + ", " + transform.rotation.z + ")");


//        Quaternion newDirection = Quaternion.FromToRotation(shipVector,headVector);
        Quaternion newDirection = transform.localRotation; 
        // Roll, Yaw, Pitch
        newDirection *= Quaternion.Euler(roll, -90, 15); // make 0,0,0 be center of players view. compensate for 2 degrees of roll drift


        //newDirection = Quaternion.Slerp(newDirection, Quaternion.Euler(transform.rotation.x, newDirection.y, newDirection.z) , Time.deltaTime * smooth *100); // dampen direction to straight ahead, no roll
//        Quaternion relativeRotation =  Quaternion.Inverse(newDirection) * Ship.rotation;


//        newDirection = Quaternion.Slerp(newDirection, Quaternion.Euler(0, Ship.rotation.y, Ship.rotation.z) , Time.deltaTime * smooth *100); // dampen direction to straight ahead, no roll


//        Vector3 angles = relativeRotation.eulerAngles;
//        Debug.Log("angle(" + angles.x + ", " + angles.y + ", " + angles.z + ")");
//        newDirection = Quaternion.Euler(angles.x, angles.y, -angles.z);
        newDirection = Ship.rotation * newDirection;
        Ship.rotation = (Quaternion.Slerp(Ship.rotation, newDirection,  Time.deltaTime * smooth));
//        Ship.rotation = (Quaternion.Slerp(Ship.rotation, Quaternion.Euler(0.0f, Ship.rotation.y,Ship.rotation.z),  Time.deltaTime * smooth));
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
        this.transform.localPosition = joystickSpawnPosition;
        this.transform.localRotation = joystickSpawnRotation;
    }
}
