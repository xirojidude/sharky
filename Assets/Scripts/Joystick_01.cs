
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
        Quaternion bodyDirection =player.GetRotation();
        Vector3 bodyVector = bodyDirection.eulerAngles;
        Debug.Log("body(" + bodyVector.x + ", " + bodyVector.y + ", " + bodyVector.z + ")");
        Vector3 shipVector = Ship.rotation.eulerAngles;
        Debug.Log("ship(" + Ship.rotation.x + ", " + Ship.rotation.y + ", " + Ship.rotation.z + ") LocalTransforn(" + transform.localEulerAngles.x + ", " + transform.localEulerAngles.y + ", " + transform.localEulerAngles.z + ")");

        Quaternion headDirection = player.GetBoneRotation( HumanBodyBones.Head );
        Vector3 headVector = headDirection.eulerAngles;
        Debug.Log("head(" + headVector.x + ", " + headVector.y + ", " + headVector.z + ")  Transforn(" + transform.rotation.x + ", " + transform.rotation.y + ", " + transform.rotation.z + ")");


//        Quaternion newDirection = Quaternion.FromToRotation(shipVector,headVector);
        Quaternion newDirection = transform.localRotation; //headDirection;
        newDirection *= Quaternion.Euler(0, -90, 0);
        Quaternion relativeRotation =  Quaternion.Inverse(newDirection) * Ship.rotation;
        Vector3 angles = relativeRotation.eulerAngles;
        Debug.Log("angle(" + angles.x + ", " + angles.y + ", " + angles.z + ")");
//        newDirection = Quaternion.Euler(angles.x, angles.y, -angles.z);
        newDirection = Ship.rotation * newDirection;
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
